"""MVP gate: allow | queue_human. Hard BLOCK out of v0."""

from __future__ import annotations

from .models import Decision, ProbeParsed


def decide(parsed: ProbeParsed) -> Decision:
    if not parsed.raw_parse_ok or parsed.would_proceed is None:
        return "queue_human"
    if parsed.uncertainty in ("high", "unknown"):
        return "queue_human"
    if parsed.would_proceed is False:
        return "queue_human"
    return "allow"
