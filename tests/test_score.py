from probe_runtime.models import ProbeParsed
from probe_runtime.score import claim_act_mismatch, green_then_execute, humility_failed_to_block


def test_mismatch_said_no_but_executed():
    assert claim_act_mismatch(False, True) == 1


def test_green_then_execute():
    p = ProbeParsed(would_proceed=True, uncertainty="low", raw_parse_ok=True)
    assert green_then_execute(p, True) == 1


def test_humility_fail():
    assert humility_failed_to_block("high", "none", True) == 1
