"""pre-tool / post-tool hooks for refund.create."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any

from .gate import decide
from .models import ProbeParsed
from .probes import sample_probe
from .score import claim_act_mismatch, green_then_execute, humility_failed_to_block
from .store import ReplayStore


def _sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def parse_probe_response(text: str) -> ProbeParsed:
    try:
        data = json.loads(text)
        unc = data.get("uncertainty", "unknown")
        if unc not in ("none", "low", "high"):
            unc = "unknown"
        return ProbeParsed(
            would_proceed=bool(data["would_proceed"]),
            uncertainty=unc,  # type: ignore[arg-type]
            raw_parse_ok=True,
            goal_one_liner=str(data.get("goal_one_liner", "")),
            rationale_one_liner=str(data.get("rationale_one_liner", "")),
        )
    except Exception:
        return ProbeParsed(would_proceed=None, uncertainty="unknown", raw_parse_ok=False)


# In-flight intents keyed by triplet_id
_PENDING: dict[str, dict[str, Any]] = {}


def pre_tool(
    *,
    agent_id: str,
    session_id: str,
    tool_name: str,
    tool_args: dict[str, Any],
    amount_minor: int,
    currency: str,
    probe_response_text: str,
    store: ReplayStore,
) -> dict[str, Any]:
    if tool_name != "refund.create":
        raise ValueError("v0 surface is refund.create only")

    probe = sample_probe()
    parsed = parse_probe_response(probe_response_text)
    decision = decide(parsed)
    triplet_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()

    _PENDING[triplet_id] = {
        "agent_id": agent_id,
        "session_id": session_id,
        "ts_utc": ts,
        "surface": {
            "tool_name": tool_name,
            "tool_args_hash": _sha256_obj(tool_args),
            "amount_minor": amount_minor,
            "currency": currency,
            "merchant_ref": tool_args.get("merchant_ref"),
        },
        "probe": probe,
        "parsed": parsed,
        "decision": decision,
        "probe_text_hash": _sha256_obj(probe.get("text", "")),
    }

    return {
        "decision": decision,
        "triplet_id": triplet_id,
        "probe_id": probe["probe_id"],
        "probe_text": probe["text"],
    }


def post_tool(
    *,
    triplet_id: str,
    executed: bool,
    provider_status: str,
    provider_ref: str | None,
    store: ReplayStore,
) -> dict[str, Any]:
    pending = _PENDING.pop(triplet_id, None)
    if pending is None:
        raise KeyError(f"unknown triplet_id: {triplet_id}")

    parsed: ProbeParsed = pending["parsed"]
    decision = pending["decision"]
    constraint = "queue_human" if decision == "queue_human" else "none"
    if decision == "queue_human" and not executed:
        # humility constrained via queue
        constrained = True
        constraint_action = "queue_human"
    elif parsed.uncertainty in ("high", "unknown") or parsed.would_proceed is False:
        constrained = not executed
        constraint_action = constraint if not executed else "none"
    else:
        constrained = False
        constraint_action = "none"

    scores = {
        "claim_act_mismatch": claim_act_mismatch(parsed.would_proceed, executed),
        "green_then_execute": green_then_execute(parsed, executed),
        "humility_failed_to_block": humility_failed_to_block(
            parsed.uncertainty, constraint_action if executed else constraint_action, executed
        ),
    }
    # Recompute humility fail with actual constraint when executed
    scores["humility_failed_to_block"] = humility_failed_to_block(
        parsed.uncertainty, "none" if executed and decision == "allow" else constraint_action, executed
    )

    record = {
        "schema_version": "sce.replay.v1",
        "triplet_id": triplet_id,
        "ts_utc": pending["ts_utc"],
        "agent_id": pending["agent_id"],
        "session_id": pending["session_id"],
        "surface": pending["surface"],
        "frame_claim": {
            "probe_id": pending["probe"]["probe_id"],
            "probe_family": pending["probe"].get("family"),
            "probe_text_hash": pending["probe_text_hash"],
            "response_text": "",  # caller may fill; not required for score
            "parsed": {
                "would_proceed": parsed.would_proceed,
                "uncertainty": parsed.uncertainty,
                "raw_parse_ok": parsed.raw_parse_ok,
            },
        },
        "frame_act": {
            "tool_name": "refund.create",
            "executed": executed,
            "blocked_by_gate": decision == "queue_human" and not executed,
            "provider_status": provider_status,
            "provider_ref": provider_ref,
        },
        "frame_humility": {
            "claimed_uncertainty": parsed.uncertainty,
            "constrained_behavior": constrained,
            "constraint_action": constraint_action,
        },
        "scores": scores,
    }
    return store.append(record)
