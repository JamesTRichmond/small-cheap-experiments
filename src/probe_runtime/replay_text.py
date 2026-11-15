"""Plain-text screen for one serialized FrameTriplet.

A payments risk analyst should be able to read the top in one pass: what the
agent said, what refund call it made, whether stated doubt stopped anything,
and a verdict. Three labeled lines follow: structural validity, integrity,
and the claim-versus-act result.

format_triplet does not recompute a hash. Its integrity line stays
"not verified" unless the caller passes a status. The JSONL command
checks the chain with hash_chain.verify_chain and record_hash, then
passes that status in. A chain whose links recompute is labeled
internally consistent, not anchored: the latest hash is not checked
against an outside copy. verify_chain returns only a bool, so the
command names a broken record itself.

It also does not score the claim against the act. PR #13 adds
claim_act_verdict on a branch that is not merged, so this file does not
import score.py. A verdict is shown only when the record or its scores
already carry claim_act_verdict or verdict. Otherwise the line is
"not assessable". A claim_act_mismatch of 0 is not read as a match.

There is one triplet-level ts_utc. Frame bodies are not given their own
timestamps, and a missing, naive, or malformed value is shown as stored.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .hash_chain import GENESIS, record_hash, verify_chain

_SCHEMA_VERSION = "sce.replay.v1"
_MISSING = object()
# Tokens from claim_act_verdict (not merged; see PR #13). Display text is
# separate: the not-assessable token is shown as the phrase "not assessable".
_VERDICT_DISPLAY = {
    "match": "match",
    "mismatch": "mismatch",
    "not_assessable": "not assessable",
    "not-assessable": "not assessable",
    "not assessable": "not assessable",
}
_VERDICT_KEYS = ("claim_act_verdict", "verdict")
# A writer who can rewrite the log can also rewrite every hash so the
# links still match. That is internal consistency, not an outside anchor.
_CHAIN_INTERNALLY_CONSISTENT = (
    "Chain integrity: internally consistent "
    "(not anchored: latest hash not checked against an outside copy)"
)


def format_triplet(
    record: Any,
    *,
    include_raw_json: bool = False,
    integrity_status: str | None = None,
) -> str:
    """Render one serialized FrameTriplet. Malformed input is a labeled line.

    integrity_status is set only by the chain command after it has checked
    the hash. Left unset, this function does not recompute a hash.
    """
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
        _verdict_line(result),
        "",
        _structural_line(record),
        (
            f"Integrity: {integrity_status}"
            if integrity_status is not None
            else _integrity_line(record)
        ),
        f"Claim vs act: {result}",
    ]
    if include_raw_json:
        lines.extend(["", _raw_json_block(record)])
    return "\n".join(lines) + "\n"


def claim_vs_act_result(record: dict[str, Any]) -> str:
    """Display a verdict the record already carries, or not assessable.

    This does not call claim_act_verdict and does not turn a zero
    claim_act_mismatch into a match. Recognized tokens are match, mismatch,
    and not_assessable (also the hyphenated and spaced spellings).
    """
    carried = _carried_verdicts(record)
    if not carried:
        return "not assessable"
    labels: list[str] = []
    for value in carried:
        if not isinstance(value, str):
            return "not assessable"
        label = _VERDICT_DISPLAY.get(value)
        if label is None:
            return "not assessable"
        labels.append(label)
    distinct = set(labels)
    if len(distinct) != 1:
        return "not assessable"
    return labels[0]


def _carried_verdicts(record: dict[str, Any]) -> list[Any]:
    found: list[Any] = []
    for key in _VERDICT_KEYS:
        if key in record:
            found.append(record[key])
    scores = record.get("scores")
    if isinstance(scores, dict):
        for key in _VERDICT_KEYS:
            if key in scores:
                found.append(scores[key])
    return found


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
            if key in _VERDICT_KEYS:
                if not isinstance(scores[key], str):
                    problems.append(f"scores.{key} is not a string")
                continue
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


def _verdict_line(result: str) -> str:
    if result == "match":
        return "Verdict: match."
    if result == "mismatch":
        return "Verdict: mismatch."
    return "Verdict: not assessable."


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


class _BadLine:
    """A JSONL slot that is not one JSON object."""

    def __init__(self, line_no: int, reason: str) -> None:
        self.line_no = line_no
        self.reason = reason


def parse_jsonl(text: str) -> list[Any]:
    """Split on ``\\n`` only, matching the store, and skip blank lines.

    A bad line becomes a labeled slot. It does not raise.
    """
    records: list[Any] = []
    for line_no, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            records.append(_BadLine(line_no, "not valid JSON"))
            continue
        if not isinstance(value, dict):
            records.append(_BadLine(line_no, "not a JSON object"))
            continue
        records.append(value)
    return records


def chain_status(records: list[Any]) -> tuple[str, list[str]]:
    """Header plus one integrity status per record.

    ``verify_chain`` returns only pass or fail. It does not say which record
    failed or why. This walk uses the public ``record_hash`` and ``GENESIS``
    to name the first break. A chain that passes is internally consistent.
    Nothing here compares the latest hash to a copy the writer cannot change.
    If that walk and ``verify_chain`` disagree, the statuses stay
    ``not verified`` rather than inventing a reason.
    """
    if not records:
        return _CHAIN_INTERNALLY_CONSISTENT, []
    broken_at, reason, labels = _walk_chain(records)
    checked = _verify_chain_bool(records)
    walk_ok = broken_at is None
    if checked is True and walk_ok:
        return _CHAIN_INTERNALLY_CONSISTENT, labels
    if checked is False and broken_at is not None:
        return f"Chain integrity: BROKEN at record {broken_at}: {reason}", labels
    if checked is None and broken_at is not None:
        # verify_chain cannot take a non-object. The break is still named.
        return f"Chain integrity: BROKEN at record {broken_at}: {reason}", labels
    return "Chain integrity: not verified", ["not verified"] * len(records)


def _verify_chain_bool(records: list[Any]) -> bool | None:
    if not all(isinstance(rec, dict) for rec in records):
        return None
    try:
        return verify_chain(records)
    except (TypeError, ValueError):
        return None


def _walk_chain(records: list[Any]) -> tuple[int | None, str | None, list[str]]:
    labels: list[str] = []
    prev = GENESIS
    broken_at: int | None = None
    reason: str | None = None
    for index, rec in enumerate(records, start=1):
        if broken_at is not None:
            labels.append("not verified")
            continue
        if isinstance(rec, _BadLine):
            broken_at = index
            reason = f"line {rec.line_no} is {rec.reason}"
            labels.append(f"BROKEN at record {index}: {reason}")
            continue
        if not isinstance(rec, dict):
            broken_at = index
            reason = "record is not a JSON object"
            labels.append(f"BROKEN at record {index}: {reason}")
            continue
        failure = _record_failure(rec, prev)
        if failure is not None:
            broken_at = index
            reason = failure
            labels.append(f"BROKEN at record {index}: {reason}")
            continue
        labels.append("verified")
        prev = rec["integrity"]["record_hash"]
    return broken_at, reason, labels


def _record_failure(record: dict[str, Any], prev: str) -> str | None:
    integrity = record.get("integrity")
    if not isinstance(integrity, dict):
        return "integrity is missing"
    if integrity.get("prev_hash") != prev:
        return "prev_hash does not match the previous record"
    body = {key: value for key, value in record.items() if key != "integrity"}
    try:
        recomputed = record_hash(body, prev)
    except (TypeError, ValueError):
        return "record_hash could not be recomputed"
    if recomputed != integrity.get("record_hash"):
        return "recomputed record_hash does not match integrity.record_hash"
    return None


def format_chain(records: list[Any], *, include_raw_json: bool = True) -> str:
    """Render a whole chain. Each record keeps the single-triplet screen."""
    header, labels = chain_status(records)
    parts = [header, f"Records: {len(records)}", ""]
    for index, (rec, label) in enumerate(zip(records, labels), start=1):
        parts.append(f"--- record {index} ---")
        if isinstance(rec, dict):
            rendered = format_triplet(
                rec, include_raw_json=include_raw_json, integrity_status=label
            )
            parts.append(rendered.rstrip("\n"))
        elif isinstance(rec, _BadLine):
            parts.append(f"Malformed record: line {rec.line_no} is {rec.reason}.")
        else:
            parts.append(
                f"Malformed record: expected a JSON object, got {type(rec).__name__}."
            )
        parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


def format_jsonl(text: str, *, include_raw_json: bool = True) -> str:
    return format_chain(parse_jsonl(text), include_raw_json=include_raw_json)


def format_chain_file(path: str | Path, *, include_raw_json: bool = True) -> str:
    return format_jsonl(Path(path).read_text(encoding="utf-8"), include_raw_json=include_raw_json)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        print(
            "usage: python -m probe_runtime.replay_text <file.jsonl>",
            file=sys.stderr,
        )
        return 2
    try:
        sys.stdout.write(format_chain_file(args[0]))
    except (OSError, UnicodeError) as exc:
        print(f"Malformed record: cannot read file ({exc}).", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
