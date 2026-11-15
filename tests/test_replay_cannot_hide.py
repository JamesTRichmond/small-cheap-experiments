"""The replay screen cannot hide a safety-relevant change.

hypothesis is not a project dependency. This is a fixed mutation list.
Records are synthetic. Currency is FAKE. No holdout probe text.

A safety-field edit must change the top screen, not only the raw JSON.
An edit to an unrelated field must leave the claim-versus-act line alone.
The three labeled lines parse back to the strings the formatter computed.
"""

from __future__ import annotations

import copy
from typing import Any

from probe_runtime.hash_chain import attach_integrity
from probe_runtime.replay_text import (
    chain_status,
    claim_vs_act_result,
    format_chain,
    format_triplet,
    labeled_lines,
)

_MISSING = object()


def _record() -> dict[str, Any]:
    return {
        "schema_version": "sce.replay.v1",
        "triplet_id": "SYNTH-HIDE-1",
        "ts_utc": "2026-01-15T12:00:00+00:00",
        "agent_id": "SYNTH-AGENT",
        "session_id": "SYNTH-SESSION",
        "surface": {
            "tool_name": "refund.create",
            "tool_args_hash": "SYNTH-ARGS",
            "amount_minor": 100,
            "currency": "FAKE",
            "merchant_ref": "SYNTH-MERCHANT",
        },
        "frame_claim": {
            "probe_id": "SYNTH-PROBE",
            "probe_family": "synthetic",
            "probe_text_hash": "SYNTH-PROBE-HASH",
            "response_text": "SYNTHETIC CLAIM A",
            "parsed": {
                "would_proceed": False,
                "uncertainty": "none",
                "raw_parse_ok": True,
            },
        },
        "frame_act": {
            "tool_name": "refund.create",
            "executed": True,
            "blocked_by_gate": False,
            "provider_status": "accepted",
            "provider_ref": "SYNTH-PROVIDER-REF",
        },
        "frame_humility": {
            "claimed_uncertainty": "none",
            "constrained_behavior": False,
            "constraint_action": "none",
        },
        "scores": {
            "claim_act_mismatch": 1,
            "claim_act_verdict": "mismatch",
            "green_then_execute": 0,
            "humility_failed_to_block": 0,
        },
        "verdict": "mismatch",
    }


def _bases() -> list[dict[str, Any]]:
    deny_refund = attach_integrity(_record())
    proceed = _record()
    proceed["frame_claim"]["parsed"]["would_proceed"] = True
    proceed["frame_act"]["executed"] = True
    proceed["scores"]["claim_act_verdict"] = "match"
    proceed["scores"]["claim_act_mismatch"] = 0
    proceed["verdict"] = "match"
    proceed_sealed = attach_integrity(proceed)
    unparsed = _record()
    unparsed["frame_claim"]["parsed"] = {
        "would_proceed": None,
        "uncertainty": "unknown",
        "raw_parse_ok": False,
    }
    unparsed["scores"].pop("claim_act_verdict")
    unparsed.pop("verdict")
    unparsed["scores"]["claim_act_mismatch"] = 0
    return [deny_refund, proceed_sealed, attach_integrity(unparsed)]


# One field at a time. These are the safety-relevant paths.
_SAFETY = (
    ("frame_claim", "parsed", "would_proceed"),
    ("frame_claim", "parsed", "raw_parse_ok"),
    ("frame_act", "tool_name"),
    ("frame_act", "executed"),
    ("frame_humility", "claimed_uncertainty"),
    ("frame_humility", "constrained_behavior"),
    ("frame_humility", "constraint_action"),
    ("scores", "claim_act_verdict"),
    ("scores", "verdict"),
    ("verdict",),
    ("ts_utc",),
    ("integrity", "prev_hash"),
    ("integrity", "record_hash"),
)

# Changing these must not move the claim-versus-act line.
_IRRELEVANT = (
    ("agent_id",),
    ("session_id",),
    ("triplet_id",),
    ("surface", "merchant_ref"),
    ("surface", "amount_minor"),
    ("surface", "currency"),
    ("surface", "tool_args_hash"),
    ("frame_claim", "response_text"),
    ("frame_claim", "probe_id"),
    ("frame_claim", "probe_family"),
    ("frame_act", "provider_status"),
    ("frame_act", "provider_ref"),
    ("frame_act", "blocked_by_gate"),
    ("scores", "claim_act_mismatch"),
    ("scores", "green_then_execute"),
    ("scores", "humility_failed_to_block"),
)


def _get(record: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return _MISSING
        current = current[key]
    return current


def _set(record: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any] | None:
    cloned = copy.deepcopy(record)
    current = cloned
    for key in path[:-1]:
        if not isinstance(current, dict) or not isinstance(current.get(key), dict):
            return None
        current = current[key]
    if not isinstance(current, dict):
        return None
    if path[-1] not in current and path[-1] not in ("verdict", "claim_act_verdict"):
        return None
    current[path[-1]] = value
    return cloned


def _alternate(value: Any) -> Any:
    if value is True:
        return False
    if value is False:
        return True
    if value is None or value is _MISSING:
        return "mismatch"
    if isinstance(value, str):
        if value in ("match", "mismatch", "not_assessable", "not assessable", "not-assessable"):
            return "match" if value != "match" else "mismatch"
        return value + "-MUT"
    if type(value) is int:
        return value + 1
    return "SYNTH-MUT"


def _top(record: dict[str, Any], *, chain: bool) -> str:
    if chain:
        rendered = format_chain([record], include_raw_json=True)
    else:
        rendered = format_triplet(record, include_raw_json=True)
    return rendered.split("\nRaw JSON:\n", 1)[0]


def _verdict_line(top: str) -> str:
    lines = [line for line in top.splitlines() if line.startswith("Claim vs act: ")]
    assert len(lines) == 1, top
    return lines[0]


def _parsed_labels(top: str) -> tuple[str, str, str]:
    validity = [line for line in top.splitlines() if line.startswith("Structural validity: ")]
    integrity = [line for line in top.splitlines() if line.startswith("Integrity: ")]
    verdict = [line for line in top.splitlines() if line.startswith("Claim vs act: ")]
    assert len(validity) == 1 and len(integrity) == 1 and len(verdict) == 1, top
    return validity[0], integrity[0], verdict[0]


def _expected_labels(record: dict[str, Any], *, chain: bool) -> tuple[str, str, str]:
    if not chain:
        return labeled_lines(record)
    _header, labels = chain_status([record])
    assert len(labels) == 1
    return labeled_lines(record, integrity_status=labels[0])


def test_safety_mutations_change_the_top_screen_not_only_the_raw_json() -> None:
    hidden: list[str] = []
    for base in _bases():
        for path in _SAFETY:
            current = _get(base, path)
            mutated = _set(base, path, _alternate(current))
            assert mutated is not None, path
            for chain in (False, True):
                before = _top(base, chain=chain)
                after = _top(mutated, chain=chain)
                status_before = _expected_labels(base, chain=chain)
                status_after = _expected_labels(mutated, chain=chain)
                if status_before != status_after:
                    assert before != after, path
                if before == after:
                    # A second copy of the verdict already on the screen is
                    # not a new fact. Any other unchanged safety edit is hidden.
                    shown = _verdict_line(before)
                    token = str(_alternate(current))
                    displayed = {
                        "match": "match",
                        "mismatch": "mismatch",
                        "not_assessable": "not assessable",
                        "not-assessable": "not assessable",
                    }.get(token, token)
                    repeated_verdict = (
                        path[-1] in ("verdict", "claim_act_verdict")
                        and shown == f"Claim vs act: {displayed}"
                    )
                    if not repeated_verdict:
                        hidden.append(f"{'chain' if chain else 'single'}:{'.'.join(path)}")
                    continue
                raw_before = format_triplet(base, include_raw_json=True).split("\nRaw JSON:\n", 1)[1]
                raw_after = format_triplet(mutated, include_raw_json=True).split("\nRaw JSON:\n", 1)[1]
                assert raw_before != raw_after
    assert hidden == [], hidden


def test_irrelevant_mutations_do_not_change_the_verdict_line() -> None:
    for base in _bases():
        before = _verdict_line(_top(base, chain=False))
        assert before == f"Claim vs act: {claim_vs_act_result(base)}"
        for path in _IRRELEVANT:
            current = _get(base, path)
            mutated = _set(base, path, _alternate(current))
            assert mutated is not None, path
            after = _verdict_line(_top(mutated, chain=False))
            assert after == before, path
            assert after == f"Claim vs act: {claim_vs_act_result(mutated)}"


def test_labeled_lines_round_trip_from_the_top_screen() -> None:
    samples = list(_bases())
    samples.append(_set(samples[0], ("ts_utc",), "not-a-timestamp"))
    samples.append(_set(samples[0], ("integrity", "record_hash"), "SYNTH-OTHER-HASH"))
    for record in samples:
        assert record is not None
        for chain in (False, True):
            top = _top(record, chain=chain)
            assert _parsed_labels(top) == _expected_labels(record, chain=chain)
            if not chain:
                full = format_triplet(record, include_raw_json=True)
                above, _raw = full.split("\nRaw JSON:\n", 1)
                assert _parsed_labels(above) == labeled_lines(record)


def test_rewritten_hash_is_on_the_single_record_screen_and_not_called_verified() -> None:
    base = _bases()[0]
    mutated = _set(base, ("integrity", "record_hash"), "SYNTH-REWRITTEN-HASH")
    assert mutated is not None
    top = _top(mutated, chain=False)
    assert "SYNTH-REWRITTEN-HASH" in top
    assert "Integrity: verified" not in top.splitlines()
    chain_top = _top(mutated, chain=True)
    assert "BROKEN at record 1" in chain_top
    assert _verdict_line(top) == _verdict_line(_top(base, chain=False))
