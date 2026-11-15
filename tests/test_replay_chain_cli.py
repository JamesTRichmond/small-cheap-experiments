"""JSONL chain replay. Synthetic records only. Currency is FAKE.

hash_chain.verify_chain returns a bool. These tests also check that the
text names the broken record, which that bool does not.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from probe_runtime.hash_chain import verify_chain
from probe_runtime.replay_text import format_chain_file, format_jsonl, format_triplet, main

GOLDEN = Path(__file__).resolve().parent / "golden" / "replay_chain"
REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def _load_jsonl(name: str) -> list[dict]:
    rows: list[dict] = []
    for line in (GOLDEN / f"{name}.jsonl").read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _render(name: str) -> str:
    return format_chain_file(GOLDEN / f"{name}.jsonl")


_CONSISTENT = (
    "Chain integrity: internally consistent "
    "(not anchored: latest hash not checked against an outside copy)\n"
)


def test_clean_chain_golden_is_internally_consistent() -> None:
    text = _render("clean")
    assert text == (GOLDEN / "clean.txt").read_text(encoding="utf-8")
    assert text.startswith(_CONSISTENT)
    assert "Chain integrity: verified" not in text
    assert verify_chain(_load_jsonl("clean")) is True
    assert text.count("Integrity: verified") == 3
    assert "Integrity: not verified" not in text
    assert "BROKEN" not in text


def test_tampered_middle_record_is_broken() -> None:
    text = _render("tampered")
    assert text == (GOLDEN / "tampered.txt").read_text(encoding="utf-8")
    assert verify_chain(_load_jsonl("tampered")) is False
    assert (
        "Chain integrity: BROKEN at record 2: "
        "recomputed record_hash does not match integrity.record_hash"
        in text
    )
    assert "Integrity: verified" in text
    assert text.count("Integrity: verified") == 1
    assert (
        "Integrity: BROKEN at record 2: "
        "recomputed record_hash does not match integrity.record_hash"
        in text
    )
    assert "Integrity: not verified" in text
    assert "amount_minor=999" in text


def test_deleted_record_breaks_the_link() -> None:
    text = _render("deleted")
    assert text == (GOLDEN / "deleted.txt").read_text(encoding="utf-8")
    assert verify_chain(_load_jsonl("deleted")) is False
    assert text.startswith("Records: 2\n") or "\nRecords: 2\n" in text
    assert (
        "Chain integrity: BROKEN at record 2: "
        "prev_hash does not match the previous record"
        in text
    )
    assert text.count("Integrity: verified") == 1


def test_cli_matches_the_clean_golden() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC)
    proc = subprocess.run(
        [sys.executable, "-m", "probe_runtime.replay_text", str(GOLDEN / "clean.jsonl")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0
    assert proc.stderr == ""
    assert proc.stdout == (GOLDEN / "clean.txt").read_text(encoding="utf-8")


def test_single_record_formatter_still_does_not_say_verified() -> None:
    sealed = _load_jsonl("clean")[0]
    assert verify_chain([sealed]) is True
    text = format_triplet(sealed)
    integrity = [line for line in text.splitlines() if line.startswith("Integrity: ")]
    assert len(integrity) == 1
    assert integrity[0].startswith("Integrity: not verified")
    assert sealed["integrity"]["record_hash"] in integrity[0]
    assert "Integrity: verified" not in text.splitlines()


def test_malformed_line_is_labeled_and_does_not_crash() -> None:
    first = (GOLDEN / "clean.jsonl").read_text(encoding="utf-8").split("\n")[0]
    text = format_jsonl(first + "\n{not-json\n")
    assert "Malformed record: line 2 is not valid JSON." in text
    assert "Chain integrity: BROKEN at record 2: line 2 is not valid JSON" in text
    assert "Integrity: verified" in text
    assert not text.startswith("Traceback")


def test_blank_lines_are_not_records() -> None:
    one = (GOLDEN / "clean.jsonl").read_text(encoding="utf-8").split("\n")[0]
    text = format_jsonl("\n" + one + "\n\n")
    assert "Records: 1\n" in text
    assert text.startswith(_CONSISTENT)


def test_empty_file_is_internally_consistent_and_not_anchored() -> None:
    assert format_jsonl("") == _CONSISTENT + "Records: 0\n"


def test_usage_and_missing_file() -> None:
    assert main([]) == 2
    code = main(["/tmp/does-not-exist-synth-replay.jsonl"])
    assert code == 1
