#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from powder.markov_utils import (
    ContinuousMarkovModel,
    StateTransition,
    FailureParameters,
    lossless_numerics,
    get_state_to_id_dict,
    fill_empty_transitions,
)
from typing import Optional
from fractions import Fraction
import math


def get_cmm(
    num_nodes: int,
    failure_params: FailureParameters,
    *,
    initial_state_dist: Optional[dict[str, lossless_numerics]] = None,
) -> ContinuousMarkovModel:
    if initial_state_dist is None:
        initial_state_dist = {"0f": Fraction(1, 1)}
    max_num_failures = int(math.ceil(num_nodes / 2)) - 1
    state_transitions = [
        StateTransition(  # 0f can only fail, no recovery
            state_name="0f",
            transition_rates=[("1f", num_nodes * failure_params.failure_rps)],
        ),
        StateTransition(  # Failed can only recover, not fail
            state_name="Failed",
            transition_rates=[("0f", failure_params.human_recovery_rps)],
        ),
        StateTransition(  # Failed - 1 transitions to Failed instead of '<num_failed>f'
            state_name=f"{max_num_failures}f",
            transition_rates=[
                (f"{max_num_failures - 1}f", failure_params.recovery_rps),
                ("Failed", (num_nodes - max_num_failures) * failure_params.failure_rps),
            ],
        ),
    ]
    for num_failures in range(1, max_num_failures):
        cur_num_working_nodes = num_nodes - num_failures
        state_transitions.append(
            StateTransition(
                state_name=f"{num_failures}f",
                transition_rates=[
                    (f"{num_failures - 1}f", failure_params.recovery_rps),
                    (
                        f"{num_failures + 1}f",
                        cur_num_working_nodes * failure_params.failure_rps,
                    ),
                ],
            )
        )
    return ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(state_transitions),
        initial_state_dist=initial_state_dist,
        state_transitions=state_transitions,
    )


def get_dr_cmm(
    num_nodes: int,
    failure_params: FailureParameters,
    *,
    initial_state_dist: Optional[dict[str, lossless_numerics]] = None,
) -> ContinuousMarkovModel:
    if initial_state_dist is None:
        initial_state_dist = {f"{num_nodes}:0": Fraction(1, 1)}
    state_transitions = [
        StateTransition(  # Failed can only recover, not fail
            state_name="Failed",
            transition_rates=[(f"{num_nodes}:0", failure_params.human_recovery_rps)],
        ),
    ]

    unexplored_states = [f"{num_nodes}:0"]
    discovered_states = set(["Failed", f"{num_nodes}:0"])
    max_failures = int(math.ceil(num_nodes / 2)) - 1
    min_updated_to_commit = num_nodes - max_failures

    while len(unexplored_states):
        cur_state = unexplored_states.pop()
        num_up_to_date = int(cur_state.split(":")[0])
        num_out_of_date = int(cur_state.split(":")[1])
        nodes_working = num_out_of_date + num_up_to_date
        transition_rates = []

        is_full_updated = num_up_to_date == nodes_working
        is_able_to_commit = num_up_to_date >= min_updated_to_commit
        is_not_live = nodes_working < min_updated_to_commit
        should_recover = nodes_working < num_nodes

        if not is_full_updated:
            transition_rates.append(
                (f"{num_up_to_date+1}:{num_out_of_date-1}", failure_params.update_rps)
            )
        if should_recover:
            if is_able_to_commit:
                transition_rates.append(
                    (
                        f"{num_up_to_date+1}:{num_out_of_date}",
                        failure_params.recovery_rps,
                    )
                )
            elif is_not_live:
                transition_rates.append(
                    (f"{num_nodes}:0", failure_params.human_recovery_rps)
                )
        if num_up_to_date > min_updated_to_commit:
            transition_rates.append(
                (
                    f"{min_updated_to_commit}:{nodes_working-min_updated_to_commit}",
                    failure_params.outdate_rps,
                )
            )
        if num_up_to_date > 1:
            transition_rates.append(
                (
                    f"{num_up_to_date-1}:{num_out_of_date}",
                    num_up_to_date * failure_params.failure_rps,
                )
            )
        if num_up_to_date == 1:
            transition_rates.append(
                (f"Failed", num_up_to_date * failure_params.failure_rps)
            )
        if num_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date-1}",
                    num_out_of_date * failure_params.failure_rps,
                )
            )

        new_states = [x[0] for x in transition_rates if x[0] not in discovered_states]
        for state in new_states:
            discovered_states.add(state)
        unexplored_states.extend(new_states)

        state_transitions.append(
            StateTransition(
                state_name=f"{num_up_to_date}:{num_out_of_date}",
                transition_rates=transition_rates,
            )
        )

    cmm = ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(state_transitions),
        initial_state_dist=initial_state_dist,
        state_transitions=state_transitions,
    )
    fill_empty_transitions(cmm)
    return cmm


def get_dr_good_bad_cmm(
    num_reliable_nodes: int,
    num_unreliable_nodes: int,
    reliable_parameters: FailureParameters,
    unreliable_parameters: FailureParameters,
    *,
    initial_state_dist: Optional[dict[str, lossless_numerics]] = None,
):
    starting_state = f"{num_reliable_nodes}:0:{num_unreliable_nodes}:0"
    if initial_state_dist is None:
        initial_state_dist = {starting_state: Fraction(1, 1)}
    overall_human_recovery_rate = max(
        reliable_parameters.human_recovery_rps, unreliable_parameters.human_recovery_rps
    )
    state_transitions = [
        StateTransition(  # Failed can only recover, not fail
            state_name="Failed",
            transition_rates=[(starting_state, overall_human_recovery_rate)],
        ),
    ]

    num_nodes = num_reliable_nodes + num_unreliable_nodes
    unexplored_states = [starting_state]
    discovered_states = set(["Failed", starting_state])
    max_failures = int(math.ceil(num_nodes / 2)) - 1
    min_updated_to_commit = num_nodes - max_failures

    while len(unexplored_states):
        cur_state = unexplored_states.pop()
        (
            num_reliable_up_to_date,
            num_reliable_out_of_date,
            num_unreliable_up_to_date,
            num_unreliable_out_of_date,
        ) = [int(x) for x in cur_state.split(":")]

        num_up_to_date = num_reliable_up_to_date + num_unreliable_up_to_date
        num_out_of_date = num_reliable_out_of_date + num_unreliable_out_of_date
        reliable_nodes_working = num_reliable_out_of_date + num_reliable_up_to_date
        unreliable_nodes_working = (
            num_unreliable_out_of_date + num_unreliable_up_to_date
        )
        nodes_working = reliable_nodes_working + unreliable_nodes_working
        transition_rates = []
        
        min_reliable_updated_to_commit = (
            min_updated_to_commit - num_unreliable_up_to_date
        )
        min_unreliable_updated_to_commit = (
            min_updated_to_commit - num_reliable_up_to_date
        )

        is_able_to_commit = num_up_to_date >= min_updated_to_commit
        is_not_live = nodes_working < min_updated_to_commit

        should_reliable_recover = reliable_nodes_working < num_reliable_nodes
        should_unreliable_recover = unreliable_nodes_working < num_unreliable_nodes

        num_ways_to_recover = int(should_reliable_recover and is_able_to_commit) + int(
            should_unreliable_recover and is_able_to_commit
        )
        # Reliable Recovery
        if should_reliable_recover and is_able_to_commit:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date + 1}:{num_reliable_out_of_date}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date}",
                    reliable_parameters.recovery_rps / num_ways_to_recover,
                )
            )
        # Unreliable Recovery
        if should_unreliable_recover and is_able_to_commit:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date}:{num_unreliable_up_to_date + 1}:{num_unreliable_out_of_date}",
                    unreliable_parameters.recovery_rps / num_ways_to_recover,
                )
            )

        # Human Recovery
        if (should_reliable_recover or should_unreliable_recover) and is_not_live:
            transition_rates.append((starting_state, overall_human_recovery_rate))

        # Reliable Update/Outdate
        if num_up_to_date > 0 and num_reliable_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date + 1}:{num_reliable_out_of_date - 1}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date}",
                    reliable_parameters.update_rps,
                )
            )
        if (
            is_able_to_commit
            and num_reliable_up_to_date > min_reliable_updated_to_commit
        ):  # pessimistic -- we could also outdate to fewer to no reliable nodes at different rates
            transition_rates.append(
                (
                    f"{min_reliable_updated_to_commit}:{reliable_nodes_working-min_reliable_updated_to_commit}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date}",
                    reliable_parameters.outdate_rps,
                )
            )

        # Unreliable Update/Outdate
        if num_up_to_date > 0 and num_unreliable_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date}:{num_unreliable_up_to_date + 1}:{num_unreliable_out_of_date - 1}",
                    unreliable_parameters.update_rps,
                )
            )
        if (
            is_able_to_commit
            and num_unreliable_up_to_date > min_unreliable_updated_to_commit
        ):  # pessimistic -- we could also outdate to fewer to no unreliable nodes at different rates
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date}:{min_unreliable_updated_to_commit}:{unreliable_nodes_working - min_unreliable_updated_to_commit}",
                    unreliable_parameters.outdate_rps,
                )
            )

        is_reliable_last_hope = (
            num_reliable_up_to_date == 1 and num_unreliable_up_to_date == 0
        )
        is_unreliable_last_hope = (
            num_reliable_up_to_date == 0 and num_unreliable_up_to_date == 1
        )

        # Reliable Failure
        if is_reliable_last_hope:
            transition_rates.append((f"Failed", reliable_parameters.failure_rps))
        elif num_reliable_up_to_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date - 1}:{num_reliable_out_of_date}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date}",
                    num_reliable_up_to_date * reliable_parameters.failure_rps,
                )
            )
        if num_reliable_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date - 1}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date}",
                    num_reliable_out_of_date * reliable_parameters.failure_rps,
                )
            )

        # Unreliable Failure
        if is_unreliable_last_hope:
            transition_rates.append((f"Failed", unreliable_parameters.failure_rps))
        elif num_unreliable_up_to_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date}:{num_unreliable_up_to_date - 1}:{num_unreliable_out_of_date}",
                    num_unreliable_up_to_date * unreliable_parameters.failure_rps,
                )
            )
        if num_unreliable_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_reliable_up_to_date}:{num_reliable_out_of_date}:{num_unreliable_up_to_date}:{num_unreliable_out_of_date - 1}",
                    num_unreliable_out_of_date * unreliable_parameters.failure_rps,
                )
            )

        new_states = [x[0] for x in transition_rates if x[0] not in discovered_states]
        for state in new_states:
            discovered_states.add(state)
        unexplored_states.extend(new_states)

        state_transitions.append(
            StateTransition(
                state_name=cur_state,
                transition_rates=transition_rates,
            )
        )

    cmm = ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(state_transitions),
        initial_state_dist=initial_state_dist,
        state_transitions=state_transitions,
    )
    fill_empty_transitions(cmm)
    return cmm


def get_dr_backup_cmm(
    num_nodes: int,
    num_backups: int,
    replica_parameters: FailureParameters,
    backup_parameters: FailureParameters,
    *,
    initial_state_dist: Optional[dict[str, lossless_numerics]] = None,
):
    # num_nodes many nodes are up to date, and num_backups many nodes are up to date
    full_functional_state_name = f"{num_nodes}:0:{num_backups}:0"
    if initial_state_dist is None:
        initial_state_dist = {full_functional_state_name: Fraction(1, 1)}
    state_transitions = [
        StateTransition(  # Failed can only recover, not fail
            state_name="Failed",
            transition_rates=[
                (full_functional_state_name, replica_parameters.human_recovery_rps)
            ],
        ),
    ]

    unexplored_states = [full_functional_state_name]
    discovered_states = set(["Failed", full_functional_state_name])
    max_failures = int(math.ceil(num_nodes / 2)) - 1
    min_updated_to_commit = num_nodes - max_failures

    while len(unexplored_states):
        cur_state = unexplored_states.pop()
        (
            num_up_to_date,
            num_out_of_date,
            num_backups_up_to_date,
            num_backups_out_of_date,
        ) = [int(x) for x in cur_state.split(":")]
        nodes_working = num_out_of_date + num_up_to_date
        backups_working = num_backups_up_to_date + num_backups_out_of_date
        transition_rates = []

        is_rsm_updated = num_up_to_date == nodes_working
        is_able_to_commit = num_up_to_date >= min_updated_to_commit
        is_backup_updated = num_backups_up_to_date == backups_working
        is_not_live = nodes_working < min_updated_to_commit
        should_rsm_recover = nodes_working < num_nodes
        should_backups_recover = backups_working < num_backups

        num_ways_to_recover = int(should_rsm_recover and is_able_to_commit) + int(
            should_backups_recover
        )
        # RSM recovery
        if should_rsm_recover:
            if is_able_to_commit:  # commit to recovery automatically
                transition_rates.append(
                    (
                        f"{num_up_to_date+1}:{num_out_of_date}:{num_backups_up_to_date}:{num_backups_out_of_date}",
                        replica_parameters.recovery_rps / num_ways_to_recover,
                    )
                )
            elif is_not_live:  # human recovery if too many nodes failed to commit
                transition_rates.append(
                    (
                        f"{num_nodes}:0:{num_backups_up_to_date}:{num_backups_out_of_date}",
                        replica_parameters.human_recovery_rps,
                    )
                )
        # backup recovery
        if should_backups_recover:  # backups can always recover
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date}:{num_backups_up_to_date+1}:{num_backups_out_of_date}",
                    backup_parameters.recovery_rps / num_ways_to_recover,
                )
            )

        # RSM update/outdate
        if not is_rsm_updated:  # If we need to update the RSM we will
            transition_rates.append(
                (
                    f"{num_up_to_date+1}:{num_out_of_date-1}:{num_backups_up_to_date}:{num_backups_out_of_date}",
                    replica_parameters.update_rps,
                )
            )
        if (
            num_up_to_date > min_updated_to_commit
        ):  # If we can commit, we might throw nodes out of date
            # optional: for i in range(min_updated_to_commit, num_up_to_date+1): add transition to outdate i nodes (\times num_backups)
            transition_rates.append(
                (
                    f"{min_updated_to_commit}:{nodes_working-min_updated_to_commit}:{num_backups_up_to_date}:{num_backups_out_of_date}",
                    replica_parameters.outdate_rps,
                )
            )

        # Backup update/outdate
        if not is_backup_updated:
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date}:{num_backups_up_to_date+1}:{num_backups_out_of_date-1}",
                    backup_parameters.update_rps,
                )
            )
        if is_able_to_commit and num_backups_up_to_date > 0:
            # optional: for i in range(0, backups_working+1): add transition to outdate i backups (\times num_up_to_date)
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date}:{0}:{backups_working}",
                    backup_parameters.outdate_rps,
                )
            )

        is_rsm_last_hope = num_up_to_date == 1 and num_backups_up_to_date == 0
        is_backup_last_hope = num_up_to_date == 0 and num_backups_up_to_date == 1

        # RSM failure
        if is_rsm_last_hope:
            transition_rates.append(
                (f"Failed", num_up_to_date * replica_parameters.failure_rps)
            )
        elif num_up_to_date > 0:
            transition_rates.append(
                (
                    f"{num_up_to_date-1}:{num_out_of_date}:{num_backups_up_to_date}:{num_backups_out_of_date}",
                    num_up_to_date * replica_parameters.failure_rps,
                )
            )
        if num_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date-1}:{num_backups_up_to_date}:{num_backups_out_of_date}",
                    num_out_of_date * replica_parameters.failure_rps,
                )
            )

        # Backup failure
        if is_backup_last_hope:
            transition_rates.append(
                (f"Failed", num_backups_up_to_date * backup_parameters.failure_rps)
            )
        elif num_backups_up_to_date > 0:
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date}:{num_backups_up_to_date-1}:{num_backups_out_of_date}",
                    num_backups_up_to_date * backup_parameters.failure_rps,
                )
            )
        if num_backups_out_of_date > 0:
            transition_rates.append(
                (
                    f"{num_up_to_date}:{num_out_of_date}:{num_backups_up_to_date}:{num_backups_out_of_date-1}",
                    num_backups_out_of_date * backup_parameters.failure_rps,
                )
            )

        new_states = [x[0] for x in transition_rates if x[0] not in discovered_states]
        for state in new_states:
            discovered_states.add(state)
        unexplored_states.extend(new_states)

        state_transitions.append(
            StateTransition(
                state_name=cur_state,
                transition_rates=transition_rates,
            )
        )

    cmm = ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(state_transitions),
        initial_state_dist=initial_state_dist,
        state_transitions=state_transitions,
    )
    fill_empty_transitions(cmm)
    return cmm
