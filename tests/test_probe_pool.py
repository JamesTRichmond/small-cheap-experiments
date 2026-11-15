"""Dev pools load from outside the repo. The sealed pool stays closed here.

Fixtures in this file are synthetic. They are not held-out probes, and an
aggregate count is not validation. Nothing in this lane blocks a refund.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from probe_runtime.probe_pool import (
    AGGREGATE_KEYS,
    BURNED_PROBE_IDS,
    BurnedProbeIdError,
    ProbePoolError,
    SealedAccessError,
    aggregate_records,
    collect_sealed_aggregate,
    load_dev_pool,
    resolve_pool_paths,
    sample_dev_probe,
)
from probe_runtime.probes import HOLD_OUT_DIR, sample_probe
from probe_runtime.sealed_aggregate import main

REPO = Path(__file__).resolve().parents[1]
SRC = str(REPO / "src")
CANARY = "SYNTHETIC-CANARY-not-a-probe"
PUBLIC_IDS = frozenset({"goal-001", "inc-001", "rev-001", "rev-002"})


def _record(probe_id: str, *, family: str = "synthetic", synthetic: bool = True) -> dict:
    return {
        "probe_id": probe_id,
        "family": family,
        "synthetic": synthetic,
        "text": CANARY,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _clear_pool_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SCE_DEV_PROBE_POOL", "SCE_SEALED_PROBE_POOL", "SCE_PROBE_CONFIG"):
        monkeypatch.delenv(name, raising=False)


def test_burned_ids_are_the_public_four():
    assert BURNED_PROBE_IDS == PUBLIC_IDS


def test_sample_probe_does_not_read_the_burned_holdout(monkeypatch, tmp_path):
    _clear_pool_env(monkeypatch)
    reads: list[Path] = []
    real_read = Path.read_text

    def spy(self, *args, **kwargs):
        if "probes" in self.as_posix():
            reads.append(self)
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", spy)
    with pytest.raises(ProbePoolError, match="no dev probe pool configured"):
        sample_probe(random.Random(0))
    assert reads == []


def test_sample_probe_rejects_the_in_repo_holdout(monkeypatch):
    _clear_pool_env(monkeypatch)
    with pytest.raises(ProbePoolError, match="inside the git repo"):
        sample_probe(random.Random(0), pool_path=HOLD_OUT_DIR)


def test_sample_probe_reads_a_synthetic_dev_pool(monkeypatch, tmp_path):
    _clear_pool_env(monkeypatch)
    path = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("synthetic-dev-1")])
    monkeypatch.setenv("SCE_DEV_PROBE_POOL", str(path))
    row = sample_probe(random.Random(0))
    assert row["probe_id"] == "synthetic-dev-1"
    assert row["synthetic"] is True
    assert row["text"] == CANARY


def test_directory_pool_is_sampled_from_synthetic_files(tmp_path):
    pool = tmp_path / "pool"
    pool.mkdir()
    _write_jsonl(pool / "b.jsonl", [_record("synthetic-b")])
    _write_jsonl(pool / "a.jsonl", [_record("synthetic-a")])
    seen = {sample_dev_probe(random.Random(i), pool_path=pool, env={})["probe_id"] for i in range(30)}
    assert seen == {"synthetic-a", "synthetic-b"}


def test_burned_id_in_a_dev_pool_raises_without_the_fixture_text(tmp_path):
    path = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("goal-001")])
    with pytest.raises(BurnedProbeIdError, match="burned probe id reused: goal-001") as exc:
        load_dev_pool(path, env={})
    assert CANARY not in str(exc.value)


def test_duplicate_synthetic_id_raises_without_the_fixture_text(tmp_path):
    path = _write_jsonl(
        tmp_path / "synthetic.jsonl",
        [_record("synthetic-dup"), _record("synthetic-dup")],
    )
    with pytest.raises(ProbePoolError, match="duplicate probe id: synthetic-dup") as exc:
        load_dev_pool(path, env={})
    assert CANARY not in str(exc.value)


def test_invalid_json_names_the_line_and_not_the_bytes(tmp_path):
    path = tmp_path / "synthetic.jsonl"
    path.write_text('{"probe_id":"synthetic-ok"}\n{not-json\n', encoding="utf-8")
    with pytest.raises(ProbePoolError, match="invalid json in probe pool at line 2") as exc:
        load_dev_pool(path, env={})
    assert "not-json" not in str(exc.value)


def test_empty_dev_pool_is_not_sampled(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n", encoding="utf-8")
    assert load_dev_pool(path, env={}) == []
    with pytest.raises(ProbePoolError, match="dev probe pool is empty"):
        sample_dev_probe(random.Random(0), pool_path=path, env={})


def test_dev_loader_does_not_open_the_sealed_root(tmp_path, monkeypatch):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    opened: list[Path] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(self)
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    with pytest.raises(SealedAccessError, match="dev loader cannot read the sealed pool"):
        load_dev_pool(sealed / "pool.jsonl", sealed=sealed, env={})
    assert opened == []


def test_symlink_into_the_repo_is_rejected(tmp_path):
    link = tmp_path / "pool.jsonl"
    link.symlink_to(HOLD_OUT_DIR / "goal_state.jsonl")
    with pytest.raises(ProbePoolError, match="inside the git repo"):
        load_dev_pool(link, env={})


def test_sealed_path_inside_the_repo_is_rejected_before_the_dev_file_is_used(tmp_path):
    dev = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("synthetic-dev-1")])
    with pytest.raises(ProbePoolError, match="sealed pool path is inside the git repo"):
        load_dev_pool(dev, sealed=HOLD_OUT_DIR, env={})


def test_config_paths_and_env_override(tmp_path):
    dev = tmp_path / "dev.jsonl"
    sealed = tmp_path / "sealed.jsonl"
    other = tmp_path / "other.jsonl"
    config = tmp_path / "pools.json"
    config.write_text(
        json.dumps(
            {
                "dev_pool": str(dev),
                "sealed_pool": str(sealed),
                "text": CANARY,
            }
        ),
        encoding="utf-8",
    )
    from_file = resolve_pool_paths({ "SCE_PROBE_CONFIG": str(config) })
    assert from_file.dev == dev
    assert from_file.sealed == sealed
    overridden = resolve_pool_paths(
        {
            "SCE_PROBE_CONFIG": str(config),
            "SCE_DEV_PROBE_POOL": str(other),
        }
    )
    assert overridden.dev == other
    assert overridden.sealed == sealed
    assert CANARY not in str(overridden)


def test_aggregate_counts_omit_ids_text_and_validation():
    aggregate = aggregate_records(
        [
            _record("synthetic-a", family="one"),
            _record("synthetic-b", family="two"),
            _record("synthetic-a", family="one"),
            _record("goal-001"),
            {"probe_id": "", "text": CANARY, "synthetic": True},
        ]
    )
    assert aggregate == {
        "probe_count": 2,
        "family_count": 2,
        "rejected_burned_ids": 1,
        "duplicate_ids": 1,
        "rejected_rows": 1,
        "synthetic": True,
        "held_out_validation": False,
        "blocks_refund": False,
    }
    blob = json.dumps(aggregate)
    assert CANARY not in blob
    assert "goal-001" not in blob
    assert "synthetic-a" not in blob
    assert "one" not in blob


def test_unmarked_records_are_not_held_out_validation():
    aggregate = aggregate_records(
        [{"probe_id": "synthetic-c", "family": "f", "text": CANARY}]
    )
    assert aggregate["synthetic"] is False
    assert aggregate["held_out_validation"] is False
    assert aggregate["blocks_refund"] is False


def test_parent_rejects_stdout_that_carries_a_fixture_field():
    leaked = json.dumps({"probe_count": 1, "text": CANARY})
    with pytest.raises(SealedAccessError, match="non-aggregate payload") as exc:
        collect_sealed_aggregate(runner=lambda: (0, leaked))
    assert CANARY not in str(exc.value)


def test_parent_rejects_a_nonzero_child_status():
    payload = json.dumps(aggregate_records([]))
    with pytest.raises(ProbePoolError, match="sealed pool rejected") as exc:
        collect_sealed_aggregate(runner=lambda: (2, payload))
    assert CANARY not in str(exc.value)


def test_child_main_writes_counts_only(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("synthetic-child-1", family="SYNTHETIC-FAMILY")])
    code = main(env={"SCE_SEALED_PROBE_POOL": str(path)})
    captured = capsys.readouterr()
    assert code == 0
    assert CANARY not in captured.out
    assert CANARY not in captured.err
    assert "synthetic-child-1" not in captured.out
    assert "SYNTHETIC-FAMILY" not in captured.out
    data = json.loads(captured.out)
    assert set(data) == AGGREGATE_KEYS
    assert data["probe_count"] == 1
    assert data["family_count"] == 1
    assert data["synthetic"] is True
    assert data["held_out_validation"] is False
    assert data["blocks_refund"] is False


def test_child_rejects_a_burned_id_without_echoing_it(tmp_path, capsys):
    path = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("goal-001")])
    code = main(env={"SCE_SEALED_PROBE_POOL": str(path)})
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == "sealed pool rejected\n"
    assert "goal-001" not in captured.out
    assert CANARY not in captured.out
    data = json.loads(captured.out)
    assert data["rejected_burned_ids"] == 1
    assert data["probe_count"] == 0
    assert data["held_out_validation"] is False


def test_child_rejects_unreadable_pool_without_echoing_bytes(tmp_path, capsys):
    path = tmp_path / "synthetic.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")
    code = main(env={"SCE_SEALED_PROBE_POOL": str(path)})
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == "sealed pool rejected\n"
    assert captured.out == ""
    assert "not-json" not in captured.err


def test_child_requires_a_sealed_path(capsys):
    code = main(env={})
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == "sealed probe pool is not configured\n"
    assert captured.out == ""


def test_child_rejects_a_sealed_path_inside_the_repo(capsys):
    code = main(env={"SCE_SEALED_PROBE_POOL": str(HOLD_OUT_DIR)})
    captured = capsys.readouterr()
    assert code == 2
    assert captured.err == "sealed pool rejected\n"
    assert captured.out == ""
    assert "goal-001" not in captured.err


def test_child_process_stdout_is_the_aggregate(tmp_path):
    path = _write_jsonl(tmp_path / "synthetic.jsonl", [_record("synthetic-proc-1")])
    env = os.environ.copy()
    env["PYTHONPATH"] = SRC
    env["SCE_SEALED_PROBE_POOL"] = str(path)
    env.pop("SCE_DEV_PROBE_POOL", None)
    env.pop("SCE_PROBE_CONFIG", None)
    proc = subprocess.run(
        [sys.executable, "-m", "probe_runtime.sealed_aggregate"],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert CANARY not in proc.stdout
    assert CANARY not in proc.stderr
    assert "synthetic-proc-1" not in proc.stdout
    data = json.loads(proc.stdout)
    assert data["probe_count"] == 1
    assert data["synthetic"] is True
    assert data["held_out_validation"] is False
    assert data["blocks_refund"] is False
