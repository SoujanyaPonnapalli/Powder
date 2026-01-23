from pathlib import Path
import sys
from fractions import Fraction

sys.path.append(str(Path(__file__).resolve().parents[1] / "notebooks"))

from utils.markov_utils import (  # noqa: E402
    ContinuousMarkovModel,
    StateTransition,
    get_initial_state_dist,
    get_state_to_id_dict,
    get_transition_matrix,
)


def test_get_state_to_id_dict_rejects_duplicates():
    transitions = [
        StateTransition(state_name="A", transition_rates=[]),
        StateTransition(state_name="A", transition_rates=[]),
    ]

    assert get_state_to_id_dict(transitions) is None


def test_transition_matrix_rows_sum_to_zero():
    transitions = [
        StateTransition(state_name="A", transition_rates=[("B", Fraction(1, 2))]),
        StateTransition(state_name="B", transition_rates=[("A", Fraction(1, 3))]),
    ]
    model = ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(transitions),
        initial_state_dist={"A": Fraction(1, 1)},
        state_transitions=transitions,
    )

    matrix = get_transition_matrix(model)

    assert matrix[0, 0] == -Fraction(1, 2)
    assert matrix[0, 1] == Fraction(1, 2)
    assert sum(matrix.row(0)) == 0


def test_initial_state_distribution_vector():
    transitions = [
        StateTransition(state_name="A", transition_rates=[]),
        StateTransition(state_name="B", transition_rates=[]),
    ]
    model = ContinuousMarkovModel(
        state_to_id=get_state_to_id_dict(transitions),
        initial_state_dist={"A": Fraction(1, 1)},
        state_transitions=transitions,
    )

    assert get_initial_state_dist(model) == [Fraction(1, 1), 0]
