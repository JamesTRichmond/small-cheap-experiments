"""Held-out micro-AMBUSH sampler. Never train production agents on these labels."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


HOLD_OUT_DIR = Path(__file__).resolve().parents[2] / "probes" / "holdout"


def load_family(family_file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not family_file.exists():
        return rows
    for line in family_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def sample_probe(rng: random.Random | None = None) -> dict[str, Any]:
    rng = rng or random.Random()
    pool: list[dict[str, Any]] = []
    for path in sorted(HOLD_OUT_DIR.glob("*.jsonl")):
        pool.extend(load_family(path))
    if not pool:
        raise RuntimeError(f"no held-out probes in {HOLD_OUT_DIR}")
    return rng.choice(pool)
