"""The public probe files stay burned.

These tests fail when a burned probe id is reused or a probe file is added.
Replay rows that only mention an id under frame_claim are not probe files.
Synthetic bytes used to provoke a failure are labeled synthetic. They are not
held-out probes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from probe_runtime.probe_hygiene import (
    BURNED_FILES,
    BURNED_PROBE_IDS,
    audit_files,
    audit_repo,
)

REPO = Path(__file__).resolve().parents[1]

# Public file bytes at main c39be52c1c0b87f7155e14c3278d54eec0e75c54.
EXPECTED_FILES = {
    "probes/holdout/goal_state.jsonl": (
        "b61625bb0be384eced62fbe3834d270dffccfc23db3bcfaccaba14573ccfad0d"
    ),
    "probes/holdout/incentive_flip.jsonl": (
        "e71b8d29678a0352c396a2f55d8222f49d57d09a2641a34f6882bb2f468ecf68"
    ),
    "probes/holdout/reversibility.jsonl": (
        "609e2fc1d550a3aab961538c02aec12081b8537e4a9d5d14ea9b944fe79857ed"
    ),
}
EXPECTED_IDS = frozenset({"goal-001", "inc-001", "rev-001", "rev-002"})

SYNTHETIC_TEXT = "SYNTHETIC probe fixture. Not a held-out probe."


def _public_bytes() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for rel, digest in EXPECTED_FILES.items():
        blob = (REPO / rel).read_bytes()
        assert hashlib.sha256(blob).hexdigest() == digest
        files[rel] = blob
    return files


def test_manifest_matches_the_public_burned_set():
    assert BURNED_FILES == EXPECTED_FILES
    assert BURNED_PROBE_IDS == EXPECTED_IDS


def test_this_repo_has_only_the_burned_probe_files():
    report = audit_repo(REPO)
    assert report.problems == ()
    assert report.ok


def test_public_bytes_are_clean_until_a_probe_file_is_added():
    files = _public_bytes()
    assert audit_files(files).ok

    files["probes/holdout/synthetic.jsonl"] = (
        json.dumps(
            {
                "probe_id": "synthetic-new",
                "synthetic": True,
                "text": SYNTHETIC_TEXT,
            }
        )
        + "\n"
    ).encode()
    report = audit_files(files)
    assert report.ok is False
    assert "probe file is not in the burned allowlist: probes/holdout/synthetic.jsonl" in report.problems
    assert SYNTHETIC_TEXT not in "\n".join(report.problems)


def test_jsonl_outside_probes_with_a_top_level_probe_id_fails():
    files = _public_bytes()
    files["notes/synthetic.jsonl"] = (
        json.dumps(
            {
                "probe_id": "synthetic-outside",
                "synthetic": True,
                "text": SYNTHETIC_TEXT,
            }
        )
        + "\n"
    ).encode()
    report = audit_files(files)
    assert "probe file is not in the burned allowlist: notes/synthetic.jsonl" in report.problems
    assert SYNTHETIC_TEXT not in "\n".join(report.problems)


def test_reused_burned_id_fails_and_the_message_omits_the_fixture_text():
    files = _public_bytes()
    files["pool/synthetic.jsonl"] = (
        json.dumps(
            {
                "probe_id": "goal-001",
                "synthetic": True,
                "text": SYNTHETIC_TEXT,
            }
        )
        + "\n"
    ).encode()
    report = audit_files(files)
    assert "burned probe id reused: goal-001" in report.problems
    assert SYNTHETIC_TEXT not in "\n".join(report.problems)


def test_changed_burned_bytes_fail():
    files = _public_bytes()
    original = files["probes/holdout/goal_state.jsonl"]
    files["probes/holdout/goal_state.jsonl"] = original[:-1] + b" "
    report = audit_files(files)
    assert "burned probe file changed: probes/holdout/goal_state.jsonl" in report.problems
    assert "would_proceed" not in "\n".join(report.problems)


def test_missing_burned_file_fails():
    files = _public_bytes()
    del files["probes/holdout/incentive_flip.jsonl"]
    report = audit_files(files)
    assert "burned probe file missing: probes/holdout/incentive_flip.jsonl" in report.problems


def test_replay_row_with_a_nested_probe_id_is_not_a_probe_file():
    files = _public_bytes()
    files["tests/golden/example.jsonl"] = (
        json.dumps(
            {
                "schema_version": "sce.replay.v1",
                "frame_claim": {"probe_id": "goal-001"},
            }
        )
        + "\n"
    ).encode()
    files["src/probe_runtime/example.py"] = b'BURNED = {"goal-001"}\n'
    assert audit_files(files).ok


def test_new_id_inside_a_burned_file_fails():
    files = _public_bytes()
    files["probes/holdout/goal_state.jsonl"] = (
        json.dumps(
            {
                "probe_id": "synthetic-added",
                "synthetic": True,
                "text": SYNTHETIC_TEXT,
            }
        )
        + "\n"
    ).encode()
    report = audit_files(files)
    joined = "\n".join(report.problems)
    assert "burned probe file changed: probes/holdout/goal_state.jsonl" in report.problems
    assert "probe id added inside a burned file: synthetic-added" in report.problems
    assert "burned probe id missing from the public files: goal-001" in report.problems
    assert SYNTHETIC_TEXT not in joined
