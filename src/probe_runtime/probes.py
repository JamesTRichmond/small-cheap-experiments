"""Dev-pool sampler.

The jsonl files under probes/holdout/ are burned. sample_probe does not read
them. Pass pool_path, or set SCE_DEV_PROBE_POOL, to a jsonl file or directory
outside this repo. The sealed pool is never read here.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


# Public burned files. Not a sample source.
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


def sample_probe(
    rng: random.Random | None = None,
    *,
    pool_path: Path | None = None,
) -> dict[str, Any]:
    """Sample one record from the dev pool.

    The in-git holdout is burned and is not consulted. A missing pool raises
    ProbePoolError. A path inside the repo, or inside the sealed pool, raises
    before those files are opened.
    """
    from .probe_pool import sample_dev_probe

    return sample_dev_probe(rng, pool_path=pool_path)
