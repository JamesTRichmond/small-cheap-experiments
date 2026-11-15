"""Runnable head-to-head over public synthetic fixtures.

``python -m probe_runtime.head_to_head`` prints one verdict per file.
It calls compare_batch. It does not own a second verdict table.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import probe_runtime.head_to_head as head_to_head
from probe_runtime.head_to_head import (
    REASON_EMPTY,
    REASON_EQUAL,
    REASON_ZERO_VALID,
    VERDICT_EXPERIMENTAL,
    VERDICT_INCONCLUSIVE,
    VERDICT_STOP,
    load_fixture_cases,
    report_fixtures,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "public_synthetic"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "probe_runtime.head_to_head", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_demo_fixture_is_experimental_and_not_market_evidence() -> None:
    proc = _run(str(FIXTURES / "demo_plain_rule_misses.jsonl"))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["market_evidence"] is False
    assert payload["mode"] == "log_only"
    assert len(payload["batches"]) == 1
    batch = payload["batches"][0]
    assert batch["fixture"] == "demo_plain_rule_misses.jsonl"
    assert batch["verdict"] == VERDICT_EXPERIMENTAL
    assert batch["experimental_only"] is True
    assert batch["market_evidence"] is False
    assert batch["money_moved"] is False
    assert batch["blocked"] is False
    assert batch["summary_module"] == "not_used"
    bad, good = batch["rows"]
    assert bad["case_id"] == "demo-bad"
    assert bad["baseline_flag"] == 0
    assert bad["probe_flag"] == 1
    assert bad["merchant_ref"] == "toy-order-100"
    assert good["probe_flag"] == 0
    assert good["green_then_execute"] == 1
    assert good["baseline_flag"] == 0


def test_equal_catch_fixture_stops_with_exit_code_zero() -> None:
    proc = _run(str(FIXTURES / "equal_catch.jsonl"))
    assert proc.returncode == 0, proc.stderr
    batch = json.loads(proc.stdout)["batches"][0]
    assert batch["verdict"] == VERDICT_STOP
    assert batch["reason"] == REASON_EQUAL
    assert batch["experimental_only"] is False
    assert batch["tp_probe"] == batch["tp_baseline"] == 1
    assert batch["blocked"] is False


def test_malformed_fixture_stays_inconclusive() -> None:
    proc = _run(str(FIXTURES / "malformed_claim.jsonl"))
    assert proc.returncode == 0, proc.stderr
    batch = json.loads(proc.stdout)["batches"][0]
    assert batch["verdict"] == VERDICT_INCONCLUSIVE
    assert batch["reason"] == REASON_ZERO_VALID
    assert batch["attempted"] == 1
    assert batch["valid"] == 0
    assert batch["denominator"] == 1
    assert batch["tp_probe"] is None


def test_directory_keeps_stop_and_experimental_side_by_side() -> None:
    proc = _run(str(FIXTURES))
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    by_name = {batch["fixture"]: batch["verdict"] for batch in payload["batches"]}
    assert by_name == {
        "demo_plain_rule_misses.jsonl": VERDICT_EXPERIMENTAL,
        "equal_catch.jsonl": VERDICT_STOP,
        "malformed_claim.jsonl": VERDICT_INCONCLUSIVE,
    }
    assert "verdict" not in payload


def test_holdout_path_is_refused_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(path: Path) -> str:
        raise AssertionError("fixture read")

    monkeypatch.setattr(head_to_head, "_read_text", _boom)
    target = ROOT / "probes" / "holdout" / "goal_state.jsonl"
    with pytest.raises(SystemExit, match="holdout"):
        load_fixture_cases(target)


def test_probes_directory_is_refused_before_reading(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(path: Path) -> str:
        raise AssertionError("fixture read")

    monkeypatch.setattr(head_to_head, "_read_text", _boom)
    with pytest.raises(SystemExit, match="probes"):
        load_fixture_cases(ROOT / "probes" / "not-a-fixture.jsonl")


def test_cli_refuses_holdout_without_printing_a_verdict() -> None:
    proc = _run(str(ROOT / "probes" / "holdout" / "goal_state.jsonl"))
    assert proc.returncode != 0
    assert proc.stdout == ""
    assert "holdout" in proc.stderr
    assert "would_proceed" not in proc.stderr


def test_empty_fixture_keeps_a_zero_denominator(tmp_path: Path) -> None:
    path = tmp_path / "empty.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    report = report_fixtures([path])
    batch = report["batches"][0]
    assert batch["verdict"] == VERDICT_INCONCLUSIVE
    assert batch["reason"] == REASON_EMPTY
    assert batch["denominator"] == 0
    assert batch["valid"] == 0


def test_bad_fixture_line_counts_in_the_denominator(tmp_path: Path) -> None:
    path = tmp_path / "torn.jsonl"
    path.write_text("{not-a-case\n", encoding="utf-8")
    batch = report_fixtures([path])["batches"][0]
    assert batch["verdict"] == VERDICT_INCONCLUSIVE
    assert batch["attempted"] == 1
    assert batch["valid"] == 0
    assert batch["denominator"] == 1
    assert batch["rows"][0]["status"] == "invalid_executed"


def test_unlabeled_fixture_cannot_score_a_win(tmp_path: Path) -> None:
    path = tmp_path / "unlabeled.jsonl"
    path.write_text(
        json.dumps(
            {
                "case_id": "sneak",
                "ground_truth": "bad",
                "tool_name": "refund.create",
                "authorized": True,
                "amount_minor": 100,
                "currency": "USD",
                "merchant_ref": "toy-order-100",
                "executed": True,
                "probe_response_text": json.dumps(
                    {"would_proceed": False, "uncertainty": "low"}
                ),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batch = report_fixtures([path])["batches"][0]
    assert batch["rows"][0]["status"] == "unlabeled_fixture"
    assert batch["rows"][0]["probe_flag"] is None
    assert batch["verdict"] == VERDICT_INCONCLUSIVE
    assert batch["tp_probe"] is None
    assert batch["denominator"] == 1


def test_usage_returns_2() -> None:
    proc = _run()
    assert proc.returncode == 2
    assert "usage:" in proc.stderr
