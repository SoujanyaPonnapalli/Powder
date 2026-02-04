from fractions import Fraction

from powder.markov_utils import FailureParameters
from powder.rsm_model_utils import get_cmm


def test_get_cmm_has_failed_state():
    params = FailureParameters(
        failure_rps=Fraction(1, 10),
        recovery_rps=Fraction(1, 2),
        human_recovery_rps=Fraction(1, 2),
        update_rps=Fraction(1, 1),
        outdate_rps=Fraction(1, 1),
    )

    model = get_cmm(3, params)

    assert "Failed" in model.state_to_id
    assert "0f" in model.state_to_id
