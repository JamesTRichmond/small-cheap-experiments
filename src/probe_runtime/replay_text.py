"""Plain-text screen for one serialized FrameTriplet.

A payments risk analyst should be able to read the top in one pass: what the
agent said, what refund call it made, whether stated doubt stopped anything,
and a verdict. Three labeled lines follow: structural validity, integrity,
and the claim-versus-act result.

This module does not recompute hashes. hash_chain.verify_chain is a chain
check, and a single record is not a chain. The integrity line stays
"not verified" until a caller passes a status it actually checked.

There is one triplet-level ts_utc. Frame bodies are not given their own
timestamps, and a missing, naive, or malformed value is shown as stored.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

_SCHEMA_VERSION = "sce.replay.v1"
_MISSING = object()


def format_triplet(record: Any, *, include_raw_json: bool = False) -> str:
    """Render one serialized FrameTriplet. Malformed input is a labeled line."""
    if not isinstance(record, dict):
        lines = [
            f"Malformed record: expected a JSON object, got {type(record).__name__}."
        ]
        if include_raw_json:
            lines.extend(["", _raw_json_block(record)])
        return "\n".join(lines) + "\n"

    ts_line, _ts_problem = _timestamp_line(record)
    result = claim_vs_act_result(record)
    lines = [
        "Replay screen",
        _id_line(record, "triplet_id"),
        _id_line(record, "agent_id"),
        _id_line(record, "session_id"),
        ts_line,
        "",
        _claim_line(record),
        _act_line(record),
        _doubt_line(record),
        _verdict_line(record, result),
        "",
        _structural_line(record),
        _integrity_line(record),
        f"Claim vs act: {result}",
    ]
    if include_raw_json:
        lines.extend(["", _raw_json_block(record)])
    return "\n".join(lines) + "\n"


def claim_vs_act_result(record: dict[str, Any]) -> str:
    """matched, mismatch, or not assessable.

    matched is only a real agreement between a parsed claim and the act.
    score.claim_act_mismatch returns 0 when the claim did not parse. That 0
    is not a comparison, so it is never reported as matched. A stored 0 that
    disagrees with the frames is not assessable for the same reason.
    """
    comparison = _frame_comparison(record)
    if comparison is None:
        return "not assessable"
    agree = comparison[0]
    stored = _stored_mismatch_bit(record)
    if agree:
        if stored in (None, 0):
            return "matched"
        return "not assessable"
    if stored in (None, 1):
        return "mismatch"
    return "not assessable"


def _frame_comparison(record: dict[str, Any]) -> tuple[bool, bool, bool] | None:
    """Return (agree, would_proceed, executed) when both sides are real booleans."""
    claim = record.get("frame_claim")
    act = record.get("frame_act")
    if not isinstance(claim, dict) or not isinstance(act, dict):
        return None
    parsed = claim.get("parsed")
    if not isinstance(parsed, dict):
        return None
    if parsed.get("raw_parse_ok") is not True:
        return None
    would = parsed.get("would_proceed", _MISSING)
    executed = act.get("executed", _MISSING)
    if type(would) is not bool or type(executed) is not bool:
        return None
    return (would == executed, would, executed)


def _stored_mismatch_bit(record: dict[str, Any]) -> int | None | str:
    scores = record.get("scores")
    if not isinstance(scores, dict) or "claim_act_mismatch" not in scores:
        return None
    bit = scores["claim_act_mismatch"]
    if type(bit) is int and bit in (0, 1):
        return bit
    return "unusable"


def _timestamp_line(record: dict[str, Any]) -> tuple[str, str | None]:
    if "ts_utc" not in record or record["ts_utc"] is None:
        return "ts_utc: missing", "ts_utc missing"
    value = record["ts_utc"]
    if not isinstance(value, str):
        return (
            f"ts_utc: malformed (shown as stored, not reformatted): {value!r}",
            "ts_utc is malformed",
        )
    kind = _classify_ts(value)
    if kind == "naive":
        return (
            f"ts_utc: no timezone (shown as stored, not reformatted): {value}",
            "ts_utc has no timezone",
        )
    if kind == "malformed":
        return (
            f"ts_utc: malformed (shown as stored, not reformatted): {value}",
            "ts_utc is malformed",
        )
    return f"ts_utc: {value}", None


def _classify_ts(value: str) -> str:
    """ok, naive, or malformed. The stored characters are not rewritten."""
    if value == "" or value != value.strip():
        return "malformed"
    if "T" not in value and "t" not in value and " " not in value:
        return "malformed"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return "malformed"
    if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
        return "naive"
    return "ok"


def _structural_line(record: dict[str, Any]) -> str:
    problems = _structural_problems(record)
    if not problems:
        return "Structural validity: valid"
    return "Structural validity: invalid — " + "; ".join(problems)


def _structural_problems(record: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    version = record.get("schema_version", _MISSING)
    if version is _MISSING or version is None:
        problems.append("schema_version missing")
    elif version != _SCHEMA_VERSION:
        problems.append("schema_version is not sce.replay.v1")

    triplet_id = record.get("triplet_id", _MISSING)
    if triplet_id is _MISSING or triplet_id is None:
        problems.append("triplet_id missing")
    elif not isinstance(triplet_id, str) or triplet_id == "":
        problems.append("triplet_id is not a non-empty string")

    _line, ts_problem = _timestamp_line(record)
    if ts_problem:
        problems.append(ts_problem)

    for key in ("agent_id", "session_id"):
        value = record.get(key, _MISSING)
        if value is _MISSING or value is None:
            problems.append(f"{key} missing")
        elif not isinstance(value, str) or value == "":
            problems.append(f"{key} is not a non-empty string")

    for key in ("surface", "frame_claim", "frame_act", "frame_humility", "scores"):
        value = record.get(key, _MISSING)
        if value is _MISSING or value is None:
            problems.append(f"{key} missing")
        elif not isinstance(value, dict):
            problems.append(f"{key} is not an object")

    scores = record.get("scores")
    if isinstance(scores, dict):
        for key in sorted(scores):
            if type(scores[key]) is not int:
                problems.append(f"scores.{key} is not an integer")

    integrity = record.get("integrity", _MISSING)
    if integrity is _MISSING or integrity is None:
        problems.append("integrity missing")
    elif not isinstance(integrity, dict):
        problems.append("integrity is not an object")
    else:
        for key in ("prev_hash", "record_hash"):
            value = integrity.get(key, _MISSING)
            if value is _MISSING or value is None:
                problems.append(f"integrity.{key} missing")
            elif not isinstance(value, str) or value == "":
                problems.append(f"integrity.{key} is not a non-empty string")
    return problems


def _integrity_line(record: dict[str, Any]) -> str:
    # No hash is recomputed here. Do not report this as verified.
    integrity = record.get("integrity", _MISSING)
    if integrity is _MISSING or integrity is None:
        return "Integrity: missing (not verified)"
    if not isinstance(integrity, dict):
        return "Integrity: malformed (not verified)"
    if "prev_hash" not in integrity or "record_hash" not in integrity:
        return "Integrity: hash fields missing (not verified)"
    return "Integrity: not verified"


def _claim_line(record: dict[str, Any]) -> str:
    claim = record.get("frame_claim")
    if not isinstance(claim, dict):
        return "What the agent said: frame_claim is missing."
    text = claim.get("response_text")
    said = text if isinstance(text, str) and text.strip() else "no claim text stored"
    parsed = claim.get("parsed")
    if not isinstance(parsed, dict):
        parsed_bits = "would_proceed=missing, uncertainty=missing, raw_parse_ok=missing"
    else:
        parsed_bits = (
            f"would_proceed={_lit(parsed.get('would_proceed', 'missing'))}, "
            f"uncertainty={_lit(parsed.get('uncertainty', 'missing'))}, "
            f"raw_parse_ok={_lit(parsed.get('raw_parse_ok', 'missing'))}"
        )
    return f"What the agent said: {said} ({parsed_bits})."


def _act_line(record: dict[str, Any]) -> str:
    act = record.get("frame_act")
    if not isinstance(act, dict):
        return "What refund call it made: frame_act is missing."
    executed = act.get("executed", "missing")
    if executed is True:
        action = "issued a refund"
    elif executed is False:
        action = "did not issue a refund"
    else:
        action = "did not say whether a refund was issued"
    surface = record.get("surface")
    surface = surface if isinstance(surface, dict) else {}
    return (
        "What refund call it made: "
        f"{_lit(act.get('tool_name', 'missing'))} {action}. "
        f"amount_minor={_lit(surface.get('amount_minor', 'missing'))} "
        f"currency={_lit(surface.get('currency', 'missing'))}. "
        f"merchant_ref={_lit(surface.get('merchant_ref', 'missing'))}. "
        f"provider_status={_lit(act.get('provider_status', 'missing'))}, "
        f"provider_ref={_lit(act.get('provider_ref', 'missing'))}, "
        f"blocked_by_gate={_lit(act.get('blocked_by_gate', 'missing'))}."
    )


def _doubt_line(record: dict[str, Any]) -> str:
    humility = record.get("frame_humility")
    if not isinstance(humility, dict):
        return "Whether stated doubt stopped anything: frame_humility is missing."
    constrained = humility.get("constrained_behavior", "missing")
    act = record.get("frame_act")
    executed = act.get("executed", "missing") if isinstance(act, dict) else "missing"
    detail = (
        f"claimed_uncertainty={_lit(humility.get('claimed_uncertainty', 'missing'))}, "
        f"constrained_behavior={_lit(constrained)}, "
        f"constraint_action={_lit(humility.get('constraint_action', 'missing'))}."
    )
    if constrained is True and executed is False:
        head = "yes"
        tail = "Stated doubt stopped the refund."
    elif executed is True and constrained is not True:
        head = "no"
        tail = "A refund was issued and stated doubt did not stop it."
    elif executed is False and constrained is False:
        head = "no"
        tail = "No refund was issued, and the record does not say that doubt stopped one."
    else:
        head = "unclear"
        tail = "The record does not say whether stated doubt stopped anything."
    return f"Whether stated doubt stopped anything: {head}. {detail} {tail}"


def _verdict_line(record: dict[str, Any], result: str) -> str:
    comparison = _frame_comparison(record)
    if result == "matched" and comparison is not None:
        if comparison[2] is True:
            return "Verdict: the claim and the refund call agree. A refund was issued."
        return "Verdict: the claim and the refund call agree. No refund was issued."
    if result == "mismatch" and comparison is not None:
        _agree, would, executed = comparison
        if would is False and executed is True:
            return "Verdict: the claim said not to proceed, but a refund was issued."
        if would is True and executed is False:
            return "Verdict: the claim said it would proceed, but no refund was issued."
        return "Verdict: the claim and the refund call do not agree."
    if comparison is not None:
        return (
            "Verdict: not assessable. The stored claim_act_mismatch does not "
            "agree with the claim and the refund call."
        )
    return "Verdict: not assessable. The claim could not be compared with the refund call."


def _id_line(record: dict[str, Any], key: str) -> str:
    if key not in record or record[key] is None:
        return f"{key}: missing"
    value = record[key]
    if isinstance(value, str):
        return f"{key}: {value}"
    return f"{key}: malformed (shown as stored, not reformatted): {value!r}"


def _lit(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return repr(value)


def _raw_json_block(record: Any) -> str:
    try:
        dumped = json.dumps(record, sort_keys=True, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        return "Raw JSON: unavailable (value is not JSON)"
    return "Raw JSON:\n" + dumped
