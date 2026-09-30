"""Diet escalation stub — webhook payload only. No tribunal in v0."""

from __future__ import annotations

from typing import Any


def should_escalate(*, humility_fails_24h: int, mismatch_rate_drop: float, manual: bool = False) -> bool:
    if manual:
        return True
    if humility_fails_24h >= 3:
        return True
    if mismatch_rate_drop >= 0.5:
        return True
    return False


def package_case(triplets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mode": "AMBUSH",
        "reason": "sce_threshold",
        "triplets": triplets,
        "note": "Hand to Diet outside this runtime. Do not run seats here.",
    }
