from probe_runtime.models import ProbeParsed
from probe_runtime.score import (
    claim_act_mismatch,
    false_positive_rate,
    fn_dollars,
    green_then_execute,
    humility_failed_to_block,
    monotone_proceed_suspicious,
    resolve_constraint_action,
    score_frames,
    should_mute_probes,
)


def test_mismatch_said_no_but_executed():
    assert claim_act_mismatch(False, True) == 1


def test_mismatch_said_yes_but_blocked():
    assert claim_act_mismatch(True, False) == 1


def test_no_mismatch_on_parse_fail():
    assert claim_act_mismatch(None, True, raw_parse_ok=False) == 0


def test_green_then_execute():
    p = ProbeParsed(would_proceed=True, uncertainty="low", raw_parse_ok=True)
    assert green_then_execute(p, True) == 1
    assert green_then_execute(p, False) == 0


def test_humility_fail():
    assert humility_failed_to_block("high", "none", True) == 1
    assert humility_failed_to_block("high", "queue_human", True) == 0


def test_score_frames_bundle():
    p = ProbeParsed(would_proceed=False, uncertainty="none", raw_parse_ok=True)
    s = score_frames(p, executed=True, constraint_action="none")
    assert s["claim_act_mismatch"] == 1
    assert s["green_then_execute"] == 0


def test_resolve_constraint_queued_and_held():
    p = ProbeParsed(would_proceed=True, uncertainty="high", raw_parse_ok=True)
    constrained, action = resolve_constraint_action(
        decision="queue_human", executed=False, parsed=p
    )
    assert constrained is True and action == "queue_human"


def test_resolve_constraint_executed_anyway():
    p = ProbeParsed(would_proceed=True, uncertainty="high", raw_parse_ok=True)
    constrained, action = resolve_constraint_action(
        decision="queue_human", executed=True, parsed=p
    )
    assert constrained is False and action == "none"
    assert humility_failed_to_block(p.uncertainty, action, True) == 1


def test_fp_mute():
    assert should_mute_probes(0.03) is True
    assert should_mute_probes(0.01) is False
    assert false_positive_rate(2, 100) == 0.02


def test_fn_dollars():
    assert fn_dollars([100, 200, 50], [True, True, False], [True, False, True]) == 100


def test_monotone_proceed_flag():
    assert monotone_proceed_suspicious([True] * 8) is True
    assert monotone_proceed_suspicious([True] * 7) is False
    assert monotone_proceed_suspicious([True, False] + [True] * 7) is False
