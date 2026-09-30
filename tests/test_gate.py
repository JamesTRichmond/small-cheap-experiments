from probe_runtime.gate import decide
from probe_runtime.models import ProbeParsed


def test_high_uncertainty_queues():
    p = ProbeParsed(would_proceed=True, uncertainty="high", raw_parse_ok=True)
    assert decide(p) == "queue_human"


def test_green_allows():
    p = ProbeParsed(would_proceed=True, uncertainty="none", raw_parse_ok=True)
    assert decide(p) == "allow"
