"""FrameTriplet and related types (sce.replay.v1)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


Uncertainty = Literal["none", "low", "high", "unknown"]
ConstraintAction = Literal["none", "refuse", "escalate", "queue_human"]
Decision = Literal["allow", "queue_human"]


@dataclass
class ProbeParsed:
    would_proceed: bool | None
    uncertainty: Uncertainty
    raw_parse_ok: bool
    goal_one_liner: str = ""
    rationale_one_liner: str = ""


@dataclass
class FrameTriplet:
    schema_version: str
    triplet_id: str
    ts_utc: str
    agent_id: str
    session_id: str
    surface: dict[str, Any]
    frame_claim: dict[str, Any]
    frame_act: dict[str, Any]
    frame_humility: dict[str, Any]
    scores: dict[str, int]
    integrity: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
