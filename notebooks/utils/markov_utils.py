#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sympy as sym
import numpy as np
from graphviz import Digraph
from dataclasses import dataclass
from fractions import Fraction
from typing import Union, Tuple, Optional
from functools import cache
from numbers import Number
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

lossless_numerics = Union[Fraction, sym.Basic]


@dataclass
class StateTransition:
    """
    no self transition allowed (does not effect modeling theoretically due to exponential distribution memorylessness)
    """

    state_name: str
    transition_rates: list[Tuple[str, lossless_numerics]]


@dataclass
class ContinuousMarkovModel:
    state_to_id: dict[str, int]
    initial_state_dist: dict[str, lossless_numerics]
    state_transitions: list[StateTransition]


@dataclass
class FailureParameters:
    """
    Global parameters useful for modeling a node's failure/recovery behavior
    """

    failure_rps: lossless_numerics
    recovery_rps: lossless_numerics
    human_recovery_rps: lossless_numerics
    update_rps: lossless_numerics
    outdate_rps: lossless_numerics

def get_state_to_id_dict(
    state_transitions: list[StateTransition],
) -> Optional[dict[str, int]]:
    """
    Algorithm:
    * state_transitions[i].cur_state_name becomes state i
    * states in initial_state_dist are ignored
    """
    state_name_to_id = {}
    for i, state_transition in enumerate(state_transitions):
        if state_transition.state_name in state_name_to_id:
            print(
                f"Error -- cannot generate transition matrix, found duplicate state [state_transition.state_name]"
            )
            return
        state_name_to_id[state_transition.state_name] = i
    return state_name_to_id

def fill_empty_transitions(markov_model: ContinuousMarkovModel):
    states_with_transitions = set(
        [x.state_name for x in markov_model.state_transitions]
    )
    states_in_total = (
        [x.state_name for x in markov_model.state_transitions]
        + [
            transition_rate[0]
            for state_transition in markov_model.state_transitions
            for transition_rate in state_transition.transition_rates
        ]
        + list(markov_model.initial_state_dist.keys())
    )
    for state in states_in_total:
        if state not in states_with_transitions:
            markov_model.state_transitions.append(
                StateTransition(state_name=state, transition_rates=[])
            )


def get_graphviz(
    markov_model: ContinuousMarkovModel,
    *,
    simple_graph: bool = False,
    show_weights: bool = False,
    node_to_group: Optional[dict[str:str]] = None,
    hide_edges_to_nodes: Optional[set[str]] = None,
    source_node: Optional[str] = None,
) -> Digraph:
    """
    TODO: Take in an argument for coloring -- i.e. live nodes green, connected to Failed node = yellow, Failed = red
    """
    if node_to_group is None:
        node_to_group = {}
    if hide_edges_to_nodes is None:
        hide_edges_to_nodes = set()

    dot = Digraph(strict="true")
    dot.attr(rankdir="LR")
    dot.attr(newrank="true")
    dot.attr(concentrate="true")
    node_to_id = {}
    for x in markov_model.state_transitions:
        if x.state_name not in node_to_id:
            node_to_id[x.state_name] = str(len(node_to_id))
        dot.node(
            node_to_id[x.state_name],
            label=x.state_name,
            group=node_to_group.get(x.state_name, None),
        )

    for state_transition in markov_model.state_transitions:
        starting_node = state_transition.state_name
        for transition in state_transition.transition_rates:
            ending_node, rate = transition
            if ending_node in hide_edges_to_nodes:
                continue
            if show_weights:
                if isinstance(rate, Number):
                    dot.edge(
                        node_to_id[starting_node],
                        node_to_id[ending_node],
                        label="{0:.4g}".format(float(rate)),
                    )
                else:
                    dot.edge(
                        node_to_id[starting_node], node_to_id[ending_node], str(rate)
                    )
            else:
                dot.edge(node_to_id[starting_node], node_to_id[ending_node])

    if simple_graph:
        return dot

    source_group_id = node_to_group.get(source_node)
    for group_id in set(node_to_group.values()):
        nodes_in_group = [k for k, v in node_to_group.items() if v == group_id]
        with dot.subgraph(name=f"cluster_{group_id}") as sub:
            sub.attr(pencolor="gray")
            sub.attr(style="dashed")
            for node in nodes_in_group:
                sub.node(node_to_id[node])
            if group_id == source_group_id:
                sub.attr(rank="min")
            else:
                sub.attr(rank="same")

    # Hack to make Failed appear rightmost
    if "Failed" in node_to_id:
        with dot.subgraph() as sub:
            sub.attr(rank="sink")
            sub.node(node_to_id["Failed"])

    return dot


def working_and_backup_grouping(markov_model: ContinuousMarkovModel) -> dict[str, str]:
    working_and_backup_grouping = {}
    for state_transition in markov_model.state_transitions:
        cur_state_name = state_transition.state_name
        numbers = cur_state_name.split(":")
        if any(not number.isnumeric() for number in numbers):
            continue
        num_backups = (
            "0" if len(numbers) < 4 else str(int(numbers[2]) + int(numbers[3]))
        )
        num_working = "0"
        for number in numbers[:2]:
            num_working = str(int(num_working) + int(number))
        working_and_backup_grouping[cur_state_name] = num_working + "_" + num_backups
    return working_and_backup_grouping

def get_wolfram_failed_id(cmm: ContinuousMarkovModel) -> int:
    return cmm.state_to_id["Failed"] + 1


def get_initial_state_dist(
    markov_model: ContinuousMarkovModel,
) -> Optional[list[lossless_numerics]]:
    if markov_model.state_to_id is None:
        return
    num_states = len(markov_model.state_to_id)
    output_vector = [0] * num_states
    for state_name, probability in markov_model.initial_state_dist.items():
        state_id = markov_model.state_to_id[state_name]
        output_vector[state_id] = probability
    return output_vector


def get_transition_matrix(markov_model: ContinuousMarkovModel) -> Optional[sym.Matrix]:
    """
    Algorithm:
    1. all_transition_rates[i].cur_state_name becomes state i
    2. For output matrix Q,
            if i != j: let Q[i][j] = be transition rate of i->j or zero if no rate specified
            otw:       let Q[i][i] = be -(Q[i][0] + Q[i][1] +...+ Q[i][i-1] + Q[i][i+1] +...+Q[i][len(all_transition_rates])
    *** 2b is required for wolfram logic
    """
    num_states = len(markov_model.state_to_id)
    output_matrix = sym.zeros(num_states, num_states)
    # Set transitions to be the rate per second
    for state_transition in markov_model.state_transitions:
        starting_state_name = state_transition.state_name
        starting_state_id = markov_model.state_to_id[starting_state_name]
        for ending_state_name, transition_rate in state_transition.transition_rates:
            ending_state_id = markov_model.state_to_id[ending_state_name]
            output_matrix[starting_state_id, ending_state_id] = transition_rate

    # Set diagonal to be negative total transition
    for i in range(output_matrix.shape[0]):
        row = output_matrix.row(i)
        output_matrix[i, i] = -np.sum(row)

    return output_matrix


def get_wolfram_markov_model(markov_model: ContinuousMarkovModel):
    """
    Convert a markov model into a wolfram language object
    """
    transition_matrix = wlexpr(
        sym.printing.mathematica.mathematica_code(get_transition_matrix(markov_model))
    )
    initial_state_dist = wlexpr(
        sym.printing.mathematica.mathematica_code(get_initial_state_dist(markov_model))
    )
    return wl.ContinuousMarkovProcess(initial_state_dist, transition_matrix)


def generate_fractions(start: Fraction, end: Fraction, num: int) -> list[Fraction]:
    """
    Generates a list of equally spaced fractions within a range.

    Args:
      start: The starting value (inclusive) as a Fraction.
      end: The ending value (inclusive) as a Fraction.
      num: The number of fractions to generate.

    Returns:
      A list of Fractions.
    """
    num = int(num)
    if num <= 0:
        return []
    if num == 1:
        return [start + end / 2]

    step = (end - start) / (num - 1)
    return [start + i * step for i in range(num)]
