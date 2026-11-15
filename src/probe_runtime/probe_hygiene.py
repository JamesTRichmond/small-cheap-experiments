"""Lock the probes that are already in public git.

The three files under ``probes/holdout/`` were published at main
``c39be52c1c0b87f7155e14c3278d54eec0e75c54``. Those bytes and their probe ids
are burned. A burned id stays burned if the file is later deleted.

``audit_repo`` reports a problem when those bytes change, when any other file
appears under ``probes/``, or when a json or jsonl file outside that allowlist
holds a top-level probe record (an object with its own ``probe_id``). A replay
row that only mentions a probe id under ``frame_claim`` is not a probe file.

This module does not sample probes and does not decide that a probe works.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# sha256 of the file bytes in public git at the commit above.
BURNED_FILES: dict[str, str] = {
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

BURNED_PROBE_IDS = frozenset({"goal-001", "inc-001", "rev-001", "rev-002"})

_SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache"}


@dataclass(frozen=True)
class HygieneReport:
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _dedupe(problems: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(problems))


def _top_level_probe_ids(blob: bytes) -> tuple[list[str], bool]:
    """Return top-level probe ids, and whether the blob could not be read.

    Only an object with its own ``probe_id`` counts. Nested ``frame_claim``
    ids are references on a replay row, not a second copy of the probe.
    """
    if not blob.strip():
        return [], False
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        return [], True

    lines = [line for line in text.split("\n") if line.strip()]
    rows: list[object] | None = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            rows = None
            break
        else:
            assert rows is not None
            rows.append(parsed)
    if rows is None:
        try:
            whole = json.loads(text)
        except json.JSONDecodeError:
            return [], True
        rows = whole if isinstance(whole, list) else [whole]

    ids: list[str] = []
    for row in rows:
        if not isinstance(row, dict) or "probe_id" not in row:
            continue
        probe_id = row["probe_id"]
        if type(probe_id) is not str or not probe_id:
            return [], True
        ids.append(probe_id)
    return ids, False


def audit_files(files: Mapping[str, bytes]) -> HygieneReport:
    """Audit an in-memory repo map of relative path to file bytes.

    Paths use forward slashes. Only ``probes/`` and ``.json`` / ``.jsonl``
    files can produce problems. Other paths are ignored so a source file that
    names a burned id is not itself a probe file.
    """
    problems: list[str] = []
    present = set(files)

    for rel, digest in BURNED_FILES.items():
        if rel not in present:
            problems.append(f"burned probe file missing: {rel}")
            continue
        if _sha256(files[rel]) != digest:
            problems.append(f"burned probe file changed: {rel}")

    for rel in sorted(present):
        if rel in BURNED_FILES:
            continue
        if rel.startswith("probes/"):
            problems.append(f"probe file is not in the burned allowlist: {rel}")

    burned_ids: list[str] = []
    outside: list[tuple[str, str]] = []
    saw_unreadable = False
    for rel in sorted(present):
        interesting = (
            rel in BURNED_FILES
            or rel.startswith("probes/")
            or rel.endswith(".json")
            or rel.endswith(".jsonl")
        )
        if not interesting:
            continue
        ids, unreadable = _top_level_probe_ids(files[rel])
        if unreadable and (
            rel in BURNED_FILES or rel.startswith("probes/") or rel.endswith(".jsonl")
        ):
            problems.append(f"unreadable probe file: {rel}")
            saw_unreadable = True
            continue
        if rel in BURNED_FILES:
            burned_ids.extend(ids)
        else:
            outside.extend((rel, probe_id) for probe_id in ids)

    burned_files_present = all(rel in present for rel in BURNED_FILES)
    if burned_files_present and not saw_unreadable:
        seen: set[str] = set()
        for probe_id in burned_ids:
            if probe_id in seen:
                problems.append(f"duplicate burned probe id: {probe_id}")
            seen.add(probe_id)
        for probe_id in sorted(BURNED_PROBE_IDS - seen):
            problems.append(f"burned probe id missing from the public files: {probe_id}")
        for probe_id in sorted(seen - BURNED_PROBE_IDS):
            problems.append(f"probe id added inside a burned file: {probe_id}")

    for rel, probe_id in outside:
        problems.append(f"probe file is not in the burned allowlist: {rel}")
        if probe_id in BURNED_PROBE_IDS:
            problems.append(f"burned probe id reused: {probe_id}")

    return HygieneReport(_dedupe(problems))


def _walk_files(root: Path) -> dict[str, bytes]:
    found: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            name
            for name in dirnames
            if name not in _SKIP_DIRS and not name.endswith(".egg-info")
        ]
        for name in filenames:
            path = Path(dirpath) / name
            rel = path.relative_to(root).as_posix()
            if not (
                rel.startswith("probes/")
                or rel.endswith(".json")
                or rel.endswith(".jsonl")
            ):
                continue
            try:
                found[rel] = path.read_bytes()
            except OSError:
                found[rel] = b"\xff"
    return found


def _tracked_probe_paths(root: Path) -> tuple[list[str], str | None]:
    if not (root / ".git").exists():
        return [], None
    proc = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z", "--", "probes"],
        capture_output=True,
    )
    if proc.returncode != 0:
        return [], "git ls-files failed"
    text = proc.stdout.decode("utf-8", errors="replace")
    return [line for line in text.split("\0") if line], None


def audit_repo(root: Path | None = None) -> HygieneReport:
    """Audit the working tree and the tracked ``probes/`` paths."""
    root = root or repo_root()
    problems = list(audit_files(_walk_files(root)).problems)
    tracked, error = _tracked_probe_paths(root)
    if error:
        problems.append(error)
    for rel in tracked:
        if rel not in BURNED_FILES:
            problems.append(f"probe file is not in the burned allowlist: {rel}")
    return HygieneReport(_dedupe(problems))


def main() -> int:
    report = audit_repo()
    for problem in report.problems:
        print(problem)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
