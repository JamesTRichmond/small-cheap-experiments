"""Fake refund records persisted by ReplayStore across a process restart.

The supported contract is the live store: append a JSON-serializable dict to an
append-only JSONL file, then reopen that path in a new interpreter. Currency in
these fixtures is FAKE. The tests call ReplayStore, not a stand-in.

Observed limits, asserted rather than papered over: a torn or corrupt line fails
closed, triplet ids are not idempotent, amounts are not validated, values that
are not JSON are rejected, and a clean process restart is not an fsync or
power-loss guarantee. In-flight refund intents are not stored here; hooks keeps
them in process memory and this file does not change hooks.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from probe_runtime.hash_chain import GENESIS
from probe_runtime.store import ReplayStore

_SRC = str(Path(__file__).resolve().parents[1] / "src")

_CHILD = r"""
import json
import sys
from pathlib import Path

from probe_runtime.store import ReplayStore

path = Path(sys.argv[1])
mode = sys.argv[2]
store = ReplayStore(path)
if mode == "read":
    rows = store._read_all()
    payload = {"verify": store.verify(), "last_hash": store.last_hash(), "rows": rows}
elif mode == "append":
    extra = json.loads(sys.argv[3])
    sealed = store.append(extra)
    payload = {
        "verify": store.verify(),
        "last_hash": store.last_hash(),
        "prev_hash": sealed["integrity"]["prev_hash"],
        "record_hash": sealed["integrity"]["record_hash"],
        "rows": store._read_all(),
    }
else:
    raise SystemExit("unknown mode")
json.dump(payload, sys.stdout)
"""


def _fake_refund(triplet_id: str, *, merchant_ref: str, amount_minor: int) -> dict:
    """Shape written for a refund triplet. The store does not interpret the fields."""
    return {
        "schema_version": "sce.replay.v1",
        "triplet_id": triplet_id,
        "ts_utc": "2026-11-07T23:00:00+00:00",
        "agent_id": "agent-fake",
        "session_id": "sess-fake",
        "surface": {
            "tool_name": "refund.create",
            "tool_args_hash": "0" * 64,
            "amount_minor": amount_minor,
            "currency": "FAKE",
            "merchant_ref": merchant_ref,
        },
        "frame_claim": {
            "probe_id": "probe-fake",
            "parsed": {"would_proceed": True, "uncertainty": "low", "raw_parse_ok": True},
        },
        "frame_act": {
            "tool_name": "refund.create",
            "executed": False,
            "blocked_by_gate": False,
            "provider_status": "not_sent",
            "provider_ref": None,
        },
        "frame_humility": {
            "claimed_uncertainty": "low",
            "constrained_behavior": False,
            "constraint_action": "none",
        },
        "scores": {"claim_act_mismatch": 0},
    }


def _restart(path: Path, mode: str, extra: dict | None = None) -> dict:
    cmd = [sys.executable, "-c", _CHILD, str(path), mode]
    if extra is not None:
        cmd.append(json.dumps(extra))
    env = os.environ.copy()
    env["PYTHONPATH"] = _SRC
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        raise AssertionError(
            f"restart helper exited {proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_fake_refund_record_persists_across_process_restart(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    store = ReplayStore(path)
    assert store.last_hash() == GENESIS
    assert store.verify() is True
    assert not path.exists()

    sealed = store.append(_fake_refund("refund-1", merchant_ref="café-订单", amount_minor=150))
    snapshot = path.read_bytes()
    got = _restart(path, "read")

    assert path.read_bytes() == snapshot
    assert got["verify"] is True
    assert got["last_hash"] == sealed["integrity"]["record_hash"]
    assert len(got["rows"]) == 1
    row = got["rows"][0]
    assert row["triplet_id"] == "refund-1"
    assert row["surface"]["tool_name"] == "refund.create"
    assert row["surface"]["currency"] == "FAKE"
    assert row["surface"]["amount_minor"] == 150
    assert row["surface"]["merchant_ref"] == "café-订单"
    assert row["frame_act"]["executed"] is False
    assert row["frame_act"]["provider_status"] == "not_sent"
    assert row["frame_act"]["provider_ref"] is None
    assert row["integrity"]["prev_hash"] == GENESIS
    assert row["integrity"]["record_hash"] == sealed["integrity"]["record_hash"]

    follow = _restart(path, "append", _fake_refund("refund-2", merchant_ref="next", amount_minor=0))
    assert follow["prev_hash"] == sealed["integrity"]["record_hash"]
    assert follow["record_hash"] == follow["last_hash"]
    assert follow["verify"] is True
    assert [r["triplet_id"] for r in follow["rows"]] == ["refund-1", "refund-2"]
    assert follow["rows"][1]["surface"]["amount_minor"] == 0

    again = _restart(path, "read")
    assert again["verify"] is True
    assert again["last_hash"] == follow["record_hash"]
    assert again["rows"][1]["integrity"]["prev_hash"] == sealed["integrity"]["record_hash"]
    assert again["rows"][0]["surface"]["merchant_ref"] == "café-订单"


@pytest.mark.parametrize(
    "merchant_ref",
    [
        "pre\u2028post",
        "pre\u2029post",
        "pre\u0085post",
        "a\u2028b\u2029c\u0085d",
        "line\nbreak",
        "carriage\rreturn",
    ],
)
def test_adversarial_merchant_ref_survives_restart(tmp_path: Path, merchant_ref: str) -> None:
    path = tmp_path / "replay.jsonl"
    sealed = ReplayStore(path).append(
        _fake_refund("sep", merchant_ref=merchant_ref, amount_minor=15)
    )

    got = _restart(path, "read")
    assert got["verify"] is True
    assert got["last_hash"] == sealed["integrity"]["record_hash"]
    assert got["rows"][0]["surface"]["merchant_ref"] == merchant_ref
    assert got["rows"][0]["surface"]["currency"] == "FAKE"

    follow = _restart(path, "append", _fake_refund("after", merchant_ref="plain", amount_minor=1))
    assert follow["prev_hash"] == sealed["integrity"]["record_hash"]
    assert follow["verify"] is True
    assert follow["rows"][0]["surface"]["merchant_ref"] == merchant_ref
    assert follow["rows"][1]["triplet_id"] == "after"


def test_crlf_file_still_verifies_across_restart(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    sealed = ReplayStore(path).append(_fake_refund("crlf", merchant_ref="m", amount_minor=3))
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))

    got = _restart(path, "read")
    assert got["verify"] is True
    assert got["last_hash"] == sealed["integrity"]["record_hash"]
    assert got["rows"][0]["surface"]["amount_minor"] == 3


def test_blank_lines_between_records_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    store = ReplayStore(path)
    first = store.append(_fake_refund("a", merchant_ref="m", amount_minor=1))
    second = store.append(_fake_refund("b", merchant_ref="m", amount_minor=2))
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    path.write_text("\n".join([lines[0], "", "  ", lines[1], ""]), encoding="utf-8")

    got = _restart(path, "read")
    assert got["verify"] is True
    assert [row["triplet_id"] for row in got["rows"]] == ["a", "b"]
    assert got["last_hash"] == second["integrity"]["record_hash"]
    assert got["rows"][1]["integrity"]["prev_hash"] == first["integrity"]["record_hash"]


def test_missing_or_blank_file_is_an_empty_chain(tmp_path: Path) -> None:
    missing = tmp_path / "nested" / "missing.jsonl"
    store = ReplayStore(missing)
    assert store.last_hash() == GENESIS
    assert store.verify() is True
    assert not missing.exists()

    blank = tmp_path / "blank.jsonl"
    blank.write_text("\n  \n\n", encoding="utf-8")
    reopened = ReplayStore(blank)
    assert reopened.last_hash() == GENESIS
    assert reopened.verify() is True
    assert reopened._read_all() == []


def test_duplicate_triplet_ids_are_both_kept(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    store = ReplayStore(path)
    store.append(_fake_refund("dup", merchant_ref="m", amount_minor=1))
    store.append(_fake_refund("dup", merchant_ref="m2", amount_minor=2))

    got = _restart(path, "read")
    assert got["verify"] is True
    kept = [
        (row["triplet_id"], row["surface"]["merchant_ref"], row["surface"]["amount_minor"])
        for row in got["rows"]
    ]
    assert kept == [("dup", "m", 1), ("dup", "m2", 2)]


def test_caller_supplied_integrity_is_replaced_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    sealed = ReplayStore(path).append(
        {
            "triplet_id": "forged",
            "surface": {
                "tool_name": "refund.create",
                "amount_minor": 5,
                "currency": "FAKE",
                "merchant_ref": "m",
            },
            "integrity": {"prev_hash": "forged", "record_hash": "forged"},
        }
    )
    assert sealed["integrity"]["prev_hash"] == GENESIS
    assert sealed["integrity"]["record_hash"] != "forged"

    got = _restart(path, "read")
    assert got["verify"] is True
    assert got["rows"][0]["integrity"] == sealed["integrity"]
    assert got["rows"][0]["surface"]["amount_minor"] == 5


def test_caller_mutation_after_append_does_not_change_restarted_record(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    record = _fake_refund("mut", merchant_ref="before", amount_minor=40)
    ReplayStore(path).append(record)
    record["surface"]["amount_minor"] = 999
    record["surface"]["merchant_ref"] = "after"

    got = _restart(path, "read")
    assert got["verify"] is True
    assert got["rows"][0]["surface"]["amount_minor"] == 40
    assert got["rows"][0]["surface"]["merchant_ref"] == "before"


def test_store_does_not_validate_fake_amounts(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    ReplayStore(path).append(_fake_refund("neg", merchant_ref="m", amount_minor=-50))

    got = _restart(path, "read")
    assert got["verify"] is True
    assert got["rows"][0]["surface"]["amount_minor"] == -50
    assert got["rows"][0]["surface"]["currency"] == "FAKE"


def test_nonserializable_payload_does_not_append(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    good = ReplayStore(path).append(_fake_refund("keep", merchant_ref="m", amount_minor=8))

    class PendingObj:
        pass

    with pytest.raises(TypeError):
        ReplayStore(path).append({"triplet_id": "bad", "parsed": PendingObj()})

    got = _restart(path, "read")
    assert got["verify"] is True
    assert [row["triplet_id"] for row in got["rows"]] == ["keep"]
    assert got["last_hash"] == good["integrity"]["record_hash"]


def test_torn_tail_fails_closed_and_keeps_prefix_bytes(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    ReplayStore(path).append(_fake_refund("ok", merchant_ref="m", amount_minor=9))
    good = path.read_bytes()
    with path.open("ab") as handle:
        handle.write(b'{"triplet_id":"torn","amount_minor":1')

    reopened = ReplayStore(path)
    with pytest.raises(json.JSONDecodeError):
        reopened.verify()
    with pytest.raises(json.JSONDecodeError):
        reopened.last_hash()
    with pytest.raises(json.JSONDecodeError):
        reopened.append(_fake_refund("next", merchant_ref="m", amount_minor=1))
    assert path.read_bytes().startswith(good)


def test_corrupt_middle_line_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    store = ReplayStore(path)
    store.append(_fake_refund("a", merchant_ref="m", amount_minor=1))
    store.append(_fake_refund("b", merchant_ref="m", amount_minor=2))
    lines = [line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]
    path.write_text(lines[0] + "\n{not-json\n" + lines[1] + "\n", encoding="utf-8")

    reopened = ReplayStore(path)
    with pytest.raises(json.JSONDecodeError):
        reopened._read_all()
    with pytest.raises(json.JSONDecodeError):
        reopened.verify()


def test_record_missing_integrity_is_not_a_valid_chain(tmp_path: Path) -> None:
    path = tmp_path / "replay.jsonl"
    path.write_text(
        json.dumps({"triplet_id": "foreign", "surface": {"currency": "FAKE", "amount_minor": 1}}) + "\n",
        encoding="utf-8",
    )
    store = ReplayStore(path)
    assert store.verify() is False
    with pytest.raises(KeyError):
        store.last_hash()
