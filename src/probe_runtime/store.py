"""Append-only JSONL replay store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hash_chain import GENESIS, attach_integrity, verify_chain


class ReplayStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        # append() delimits records with "\n" only. str.splitlines() also
        # breaks on U+2028, U+2029, and U+0085, which json.dumps leaves raw
        # inside strings, so a sealed record could not be read back.
        for line in self.path.read_text(encoding="utf-8").split("\n"):
            if line.strip():
                out.append(json.loads(line))
        return out

    def last_hash(self) -> str:
        rows = self._read_all()
        if not rows:
            return GENESIS
        return rows[-1]["integrity"]["record_hash"]

    def append(self, record: dict[str, Any]) -> dict[str, Any]:
        sealed = attach_integrity(record, self.last_hash())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sealed, ensure_ascii=False) + "\n")
        return sealed

    def verify(self) -> bool:
        return verify_chain(self._read_all())
