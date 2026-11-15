"""pre-tool / post-tool hooks for refund.create."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gate import decide
from .models import ProbeParsed
from .probes import sample_probe
from .score import resolve_constraint_action, score_frames
from .store import ReplayStore

# Public response schema. Annotation keywords carry no constraint.
_SCHEMA_ANNOTATIONS = {"$schema", "$id", "title", "description", "examples", "default"}
_SCHEMA_CONSTRAINTS = {"type", "enum", "properties", "required", "additionalProperties"}
_PROBE_RESPONSE_SCHEMA = (
    Path(__file__).resolve().parents[2] / "schemas" / "probe_response.schema.json"
)


def _sha256_obj(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _unassessable() -> ProbeParsed:
    return ProbeParsed(would_proceed=None, uncertainty="unknown", raw_parse_ok=False)


def _json_type_ok(value: Any, expected: str) -> bool:
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "string":
        return isinstance(value, str)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, float) or (
            isinstance(value, int) and not isinstance(value, bool)
        )
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported JSON Schema type {expected}")


def _validate_json_schema(data: Any, schema: dict[str, Any]) -> None:
    """Apply the keywords in the public probe-response schema. No extra deps."""
    unknown = set(schema) - _SCHEMA_ANNOTATIONS - _SCHEMA_CONSTRAINTS
    if unknown:
        raise ValueError(f"unsupported schema keywords: {sorted(unknown)}")
    expected = schema.get("type")
    if isinstance(expected, str):
        if not _json_type_ok(data, expected):
            raise ValueError("schema type")
    elif expected is not None:
        raise ValueError("unsupported schema type form")
    if "enum" in schema and data not in schema["enum"]:
        raise ValueError("schema enum")
    if expected != "object":
        return
    if schema.get("additionalProperties") is not False:
        raise ValueError("schema additionalProperties is not false")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict) or not isinstance(required, list):
        raise ValueError("schema object keywords")
    for key in required:
        if key not in data:
            raise ValueError(f"missing required {key}")
    for key in data:
        if key not in properties:
            raise ValueError(f"extra property {key}")
    for key, prop_schema in properties.items():
        if key in data:
            if not isinstance(prop_schema, dict):
                raise ValueError("schema property")
            _validate_json_schema(data[key], prop_schema)


def _probe_response_schema() -> dict[str, Any]:
    loaded = json.loads(_PROBE_RESPONSE_SCHEMA.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("probe response schema is not an object")
    return loaded


def parse_probe_response(text: str) -> ProbeParsed:
    """Parse text against schemas/probe_response.schema.json.

    A successful parse keeps an actual JSON boolean and an uncertainty value
    from the schema enum. It does not coerce with bool(). Strings, numbers,
    null, containers, missing required fields, and extra fields are invalid.

    Invalid text returns the same shape as before: would_proceed None,
    uncertainty "unknown", raw_parse_ok False. That row is not assessable and
    is not a proceed claim. Score functions are unchanged: for the same valid
    parsed claim and executed flag they still receive the same ProbeParsed.
    """
    try:
        if not isinstance(text, str):
            raise TypeError("probe response text must be a str")
        data = json.loads(text)
        _validate_json_schema(data, _probe_response_schema())
    except Exception:
        return _unassessable()
    return ProbeParsed(
        would_proceed=data["would_proceed"],
        uncertainty=data["uncertainty"],
        raw_parse_ok=True,
        goal_one_liner=data["goal_one_liner"] if "goal_one_liner" in data else "",
        rationale_one_liner=(
            data["rationale_one_liner"] if "rationale_one_liner" in data else ""
        ),
    )


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
    constrained, constraint_action = resolve_constraint_action(
        decision=decision, executed=executed, parsed=parsed
    )
    scores = score_frames(parsed, executed, constraint_action)

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


# Legacy pre_tool samples a probe after it already has probe_response_text.
# That is not evidence the response was elicited for the sampled probe.
PAIRING_UNASSESSABLE = "unassessable"


def call_refund_log_only(
    refund: Callable[[], Any],
    /,
    *,
    agent_id: str,
    session_id: str,
    tool_args: dict[str, Any],
    amount_minor: int,
    currency: str,
    probe_response_text: str,
    store: Any,
    provider_status: str,
    provider_ref: str | None = None,
    trace: dict[str, Any] | None = None,
) -> Any:
    """Call refund once. Observation does not veto or replace the outcome.

    pre_tool and post_tool still raise when called directly. This wrapper
    catches those telemetry failures so the refund return value, or the
    refund's own exception, is what the caller sees.

    The gate decision is recorded on trace and is not an input to refund.
    probe_response_text is pre-supplied. pre_tool may sample a different
    probe. trace["pairing"] is therefore "unassessable". raw_parse_ok is
    parse validity, not provenance.

    Signature:
        call_refund_log_only(refund, /, *, agent_id, session_id, tool_args,
        amount_minor, currency, probe_response_text, store, provider_status,
        provider_ref=None, trace=None) -> refund()
    """
    observed: dict[str, Any] | None = None
    pre_error: Exception | None = None
    try:
        observed = pre_tool(
            agent_id=agent_id,
            session_id=session_id,
            tool_name="refund.create",
            tool_args=tool_args,
            amount_minor=amount_minor,
            currency=currency,
            probe_response_text=probe_response_text,
            store=store,
        )
    except Exception as exc:
        pre_error = exc

    refund_error: Exception | None = None
    result: Any = None
    try:
        result = refund()
    except Exception as exc:
        refund_error = exc

    post_error: Exception | None = None
    record: dict[str, Any] | None = None
    if observed is not None:
        try:
            record = post_tool(
                triplet_id=observed["triplet_id"],
                executed=refund_error is None,
                provider_status=provider_status,
                provider_ref=provider_ref,
                store=store,
            )
        except Exception as exc:
            post_error = exc

    if trace is not None:
        trace.clear()
        trace["decision"] = None if observed is None else observed["decision"]
        trace["triplet_id"] = None if observed is None else observed["triplet_id"]
        trace["pre_error"] = pre_error
        trace["post_error"] = post_error
        trace["record"] = record
        trace["pairing"] = PAIRING_UNASSESSABLE

    if refund_error is not None:
        raise refund_error
    return result
