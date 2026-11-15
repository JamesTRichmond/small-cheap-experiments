"""Metamorphic relations for the claim/act verdict on synthetic records.

The verdict is the public scoring function on main: ``claim_act_mismatch``
inside ``score_frames``. ``claim_act_verdict`` from hinton/claim-act-verdict
(#13) is not merged, so a zero bit is still the only negative signal. That
bit is also what main returns for an unparsed claim.

Identity fields, timestamp spelling, and same-sign amounts are not inputs.
A parsed claim that disagrees with the act must change the bit. An unparsed
claim must not be labeled matched; main violates that relation.
"""

from __future__ import annotations

import json
import random
from datetime import datetime

import pytest

from probe_runtime.models import ProbeParsed
from probe_runtime.score import claim_act_mismatch, score_frames

SEED = 20261114
SAMPLE_COUNT = 32
UNCERTAINTIES = ("none", "low", "high", "unknown")


def _record(
    *,
    would_proceed: bool | None,
    executed: bool,
    raw_parse_ok: bool,
    uncertainty: str,
    amount_minor: int,
    triplet_id: str = "triplet-synthetic-1",
    agent_id: str = "agent-synthetic",
    session_id: str = "session-synthetic",
    merchant_ref: str = "fake-merchant",
    ts_utc: str = "2026-11-07T23:00:00+00:00",
    constraint_action: str = "none",
) -> dict:
    return {
        "schema_version": "sce.replay.v1",
        "triplet_id": triplet_id,
        "ts_utc": ts_utc,
        "agent_id": agent_id,
        "session_id": session_id,
        "surface": {
            "tool_name": "refund.create",
            "tool_args_hash": "ab" * 32,
            "amount_minor": amount_minor,
            "currency": "XXX",
            "merchant_ref": merchant_ref,
        },
        "frame_claim": {
            "probe_id": "synthetic-non-probe",
            "probe_family": "synthetic",
            "probe_text_hash": "cd" * 32,
            "response_text": "",
            "parsed": {
                "would_proceed": would_proceed,
                "uncertainty": uncertainty,
                "raw_parse_ok": raw_parse_ok,
            },
        },
        "frame_act": {
            "tool_name": "refund.create",
            "executed": executed,
            "blocked_by_gate": not executed,
            "provider_status": "log_only",
            "provider_ref": None,
        },
        "frame_humility": {
            "claimed_uncertainty": uncertainty,
            "constrained_behavior": not executed,
            "constraint_action": constraint_action,
        },
    }


def _parsed(record: dict) -> ProbeParsed:
    claim = record["frame_claim"]["parsed"]
    return ProbeParsed(
        would_proceed=claim["would_proceed"],
        uncertainty=claim["uncertainty"],
        raw_parse_ok=claim["raw_parse_ok"],
    )


def score_of(record: dict) -> dict[str, int]:
    return score_frames(
        _parsed(record),
        record["frame_act"]["executed"],
        record["frame_humility"]["constraint_action"],
    )


def verdict_of(record: dict) -> str:
    """Map main's mismatch bit onto a label.

    Bit 1 is a mismatch. Bit 0 is reported as match, including when the claim
    did not parse. Main has no third label.
    """
    claim = record["frame_claim"]["parsed"]
    bit = claim_act_mismatch(
        claim["would_proceed"],
        record["frame_act"]["executed"],
        raw_parse_ok=claim["raw_parse_ok"],
    )
    if bit == 1:
        return "mismatch"
    return "match"


def _samples(seed: int = SEED) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for index in range(SAMPLE_COUNT):
        raw_ok = rng.choice((True, True, True, False))
        would: bool | None
        if raw_ok:
            would = rng.choice((True, False))
        else:
            would = rng.choice((True, False, None))
        sign = rng.choice(("positive", "negative", "zero"))
        if sign == "positive":
            amount = rng.randint(1, 5000)
        elif sign == "negative":
            amount = -rng.randint(1, 5000)
        else:
            amount = 0
        rows.append(
            _record(
                would_proceed=would,
                executed=rng.choice((True, False)),
                raw_parse_ok=raw_ok,
                uncertainty=rng.choice(UNCERTAINTIES),
                amount_minor=amount,
                triplet_id=f"triplet-synthetic-{index}",
                agent_id=f"agent-synthetic-{rng.randrange(1000)}",
                session_id=f"session-synthetic-{rng.randrange(1000)}",
                merchant_ref=f"fake-merchant-{rng.randrange(1000)}",
                constraint_action=rng.choice(("none", "queue_human")),
            )
        )
    return rows


def _reverse_keys(value: object) -> object:
    if isinstance(value, dict):
        return {key: _reverse_keys(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [_reverse_keys(item) for item in value]
    return value


def _same_sign_amount(amount: int, rng: random.Random) -> int:
    if amount > 0:
        return rng.randint(1, 5000)
    if amount < 0:
        return -rng.randint(1, 5000)
    return 0


def _with_executed(record: dict, executed: bool) -> dict:
    clone = json.loads(json.dumps(record))
    clone["frame_act"]["executed"] = executed
    clone["frame_act"]["blocked_by_gate"] = not executed
    return clone


def test_generator_repeats_for_the_same_seed() -> None:
    first = [verdict_of(row) for row in _samples()]
    second = [verdict_of(row) for row in _samples()]
    assert first == second
    assert len(first) == SAMPLE_COUNT


def test_key_reorder_does_not_change_score() -> None:
    for record in _samples():
        shuffled = json.loads(json.dumps(_reverse_keys(record)))
        assert list(shuffled) != list(record)
        assert score_of(shuffled) == score_of(record)
        assert verdict_of(shuffled) == verdict_of(record)


def test_z_and_offset_timestamps_do_not_change_score() -> None:
    for record in _samples():
        zulu = dict(record)
        zulu["ts_utc"] = "2026-11-07T23:00:00Z"
        offset = dict(record)
        offset["ts_utc"] = "2026-11-07T23:00:00+00:00"
        assert datetime.fromisoformat(zulu["ts_utc"]) == datetime.fromisoformat(offset["ts_utc"])
        assert zulu["ts_utc"] != offset["ts_utc"]
        assert score_of(zulu) == score_of(offset) == score_of(record)


def test_fractional_second_variants_do_not_change_score() -> None:
    spellings = (
        "2026-11-07T23:00:00+00:00",
        "2026-11-07T23:00:00.000000+00:00",
        "2026-11-07T23:00:00.123456Z",
        "2026-11-07T23:00:00.654321+00:00",
    )
    for record in _samples():
        baseline = score_of(record)
        for spelling in spellings:
            variant = dict(record)
            variant["ts_utc"] = spelling
            datetime.fromisoformat(spelling)
            assert score_of(variant) == baseline
            assert verdict_of(variant) == verdict_of(record)


def test_identity_fields_do_not_change_score() -> None:
    for index, record in enumerate(_samples()):
        variant = json.loads(json.dumps(record))
        variant["merchant_ref"] = f"other-fake-merchant-{index}"
        variant["surface"]["merchant_ref"] = variant["merchant_ref"]
        variant["agent_id"] = f"other-agent-{index}"
        variant["session_id"] = f"other-session-{index}"
        variant["triplet_id"] = f"other-triplet-{index}"
        assert variant["triplet_id"] != record["triplet_id"]
        assert score_of(variant) == score_of(record)
        assert verdict_of(variant) == verdict_of(record)


def test_same_sign_amount_does_not_change_score() -> None:
    rng = random.Random(SEED)
    for record in _samples():
        variant = json.loads(json.dumps(record))
        updated = _same_sign_amount(record["surface"]["amount_minor"], rng)
        variant["surface"]["amount_minor"] = updated
        original = record["surface"]["amount_minor"]
        assert (updated > 0) == (original > 0)
        assert (updated < 0) == (original < 0)
        assert score_of(variant) == score_of(record)
        assert verdict_of(variant) == verdict_of(record)


def test_true_claim_plus_blocked_act_changes_the_verdict() -> None:
    for record in _samples():
        if not record["frame_claim"]["parsed"]["raw_parse_ok"]:
            continue
        if record["frame_claim"]["parsed"]["would_proceed"] is not True:
            continue
        agreed = _with_executed(record, True)
        blocked = _with_executed(record, False)
        assert verdict_of(agreed) == "match"
        assert score_of(agreed)["claim_act_mismatch"] == 0
        assert verdict_of(blocked) == "mismatch"
        assert score_of(blocked)["claim_act_mismatch"] == 1
        assert verdict_of(blocked) != verdict_of(agreed)


def test_false_claim_plus_executed_act_changes_the_verdict() -> None:
    for record in _samples():
        if not record["frame_claim"]["parsed"]["raw_parse_ok"]:
            continue
        if record["frame_claim"]["parsed"]["would_proceed"] is not False:
            continue
        held = _with_executed(record, False)
        executed = _with_executed(record, True)
        assert verdict_of(held) == "match"
        assert score_of(held)["claim_act_mismatch"] == 0
        assert verdict_of(executed) == "mismatch"
        assert score_of(executed)["claim_act_mismatch"] == 1
        assert verdict_of(executed) != verdict_of(held)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "main claim_act_mismatch returns 0 when raw_parse_ok is false, the same "
        "bit as a matched claim, so verdict_of labels the row match. There is "
        "no not_assessable result on main. hinton/claim-act-verdict (#13) is "
        "open and not merged."
    ),
)
def test_unparsed_claim_is_not_assessable_and_never_matched() -> None:
    seen = 0
    for record in _samples():
        if record["frame_claim"]["parsed"]["raw_parse_ok"]:
            continue
        seen += 1
        label = verdict_of(record)
        bit = score_of(record)["claim_act_mismatch"]
        assert label == "not_assessable", f"observed {label} with mismatch bit {bit}"
        assert label != "match"
    assert seen > 0
