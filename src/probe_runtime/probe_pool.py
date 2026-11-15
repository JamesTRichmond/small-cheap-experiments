"""Load a dev probe pool, and keep the sealed pool out of this process.

Pools live outside the git tree. ``SCE_DEV_PROBE_POOL`` and
``SCE_SEALED_PROBE_POOL`` are paths. ``SCE_PROBE_CONFIG`` may point at a JSON
object with ``dev_pool`` and ``sealed_pool``; an environment variable wins
over the file. Unknown config keys are ignored. The config file is not a pool.

The four public ids (``goal-001``, ``inc-001``, ``rev-001``, ``rev-002``) are
burned. The dev loader refuses them, refuses any path inside this repo, and
refuses the sealed path without opening it.

Only ``python -m probe_runtime.sealed_aggregate`` reads the sealed pool. The
parent accepts that child's stdout when it is the aggregate object and nothing
else. ``held_out_validation`` is always false: a count is not a measurement.
``blocks_refund`` is always false.

Synthetic records are whatever the caller marked with ``synthetic: true``.
This module does not treat them as held-out probes.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

BURNED_PROBE_IDS = frozenset({"goal-001", "inc-001", "rev-001", "rev-002"})

DEV_POOL_ENV = "SCE_DEV_PROBE_POOL"
SEALED_POOL_ENV = "SCE_SEALED_PROBE_POOL"
CONFIG_ENV = "SCE_PROBE_CONFIG"

AGGREGATE_KEYS = frozenset(
    {
        "probe_count",
        "family_count",
        "rejected_burned_ids",
        "duplicate_ids",
        "rejected_rows",
        "synthetic",
        "held_out_validation",
        "blocks_refund",
    }
)

_COUNT_KEYS = (
    "probe_count",
    "family_count",
    "rejected_burned_ids",
    "duplicate_ids",
    "rejected_rows",
)


class ProbePoolError(RuntimeError):
    """The pool path, config, or records cannot be used."""


class SealedAccessError(ProbePoolError):
    """This process tried to read the sealed pool, or the child leaked fields."""


class BurnedProbeIdError(ProbePoolError):
    """A dev record reused a probe id that is already in public git."""


@dataclass(frozen=True)
class PoolPaths:
    dev: Path | None
    sealed: Path | None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_pool_paths(env: Mapping[str, str] | None = None) -> PoolPaths:
    env = os.environ if env is None else env
    dev = env.get(DEV_POOL_ENV) or None
    sealed = env.get(SEALED_POOL_ENV) or None
    config_path = env.get(CONFIG_ENV) or None
    if config_path:
        file_dev, file_sealed = _read_config(Path(config_path))
        dev = dev or file_dev
        sealed = sealed or file_sealed
    return PoolPaths(Path(dev) if dev else None, Path(sealed) if sealed else None)


def _read_config(path: Path) -> tuple[str | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise ProbePoolError("SCE_PROBE_CONFIG could not be read") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        raise ProbePoolError("SCE_PROBE_CONFIG is not valid JSON") from None
    if not isinstance(data, dict):
        raise ProbePoolError("SCE_PROBE_CONFIG must be a JSON object")
    dev = data.get("dev_pool")
    sealed = data.get("sealed_pool")
    if dev is not None and type(dev) is not str:
        raise ProbePoolError("SCE_PROBE_CONFIG dev_pool must be a string")
    if sealed is not None and type(sealed) is not str:
        raise ProbePoolError("SCE_PROBE_CONFIG sealed_pool must be a string")
    return dev or None, sealed or None


def read_probe_records(path: Path) -> list[dict[str, Any]]:
    """Read jsonl records. Callers apply the path policy before this."""
    if path.is_dir():
        files = sorted(item for item in path.glob("*.jsonl") if item.is_file())
    elif path.is_file():
        files = [path]
    else:
        raise ProbePoolError("probe pool path does not exist")
    records: list[dict[str, Any]] = []
    for file in files:
        records.extend(_read_jsonl(file))
    return records


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        raise ProbePoolError("probe pool could not be read") from None
    except UnicodeDecodeError:
        raise ProbePoolError("probe pool is not utf-8 text") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ProbePoolError(
                f"invalid json in probe pool at line {line_number}"
            ) from None
        if not isinstance(row, dict):
            raise ProbePoolError(f"probe pool line {line_number} is not an object")
        rows.append(row)
    return rows


def load_dev_pool(
    path: Path,
    *,
    repo: Path | None = None,
    sealed: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Read a dev pool from outside the repo. Does not open the sealed path."""
    repo = (repo or repo_root()).resolve()
    target = Path(path)
    if sealed is None and env is not None:
        sealed = resolve_pool_paths(env).sealed
    elif sealed is None and env is None:
        sealed = resolve_pool_paths(os.environ).sealed
    if sealed is not None:
        sealed_path = Path(sealed)
        if _within(sealed_path, repo):
            raise ProbePoolError("sealed pool path is inside the git repo")
        if _within(target, sealed_path):
            raise SealedAccessError("dev loader cannot read the sealed pool")
    if _within(target, repo):
        raise ProbePoolError(
            "probe pool path is inside the git repo; public probes are burned "
            "and are not a sample source"
        )
    rows = read_probe_records(target.resolve())
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        probe_id = row.get("probe_id")
        if type(probe_id) is not str or not probe_id.strip():
            raise ProbePoolError(f"probe record {index} is missing a probe_id string")
        if probe_id in BURNED_PROBE_IDS:
            raise BurnedProbeIdError(f"burned probe id reused: {probe_id}")
        text = row.get("text")
        if type(text) is not str or not text.strip():
            raise ProbePoolError(f"probe record {index} is missing probe text")
        if probe_id in seen:
            raise ProbePoolError(f"duplicate probe id: {probe_id}")
        seen.add(probe_id)
    return rows


def sample_dev_probe(
    rng: random.Random | None = None,
    *,
    pool_path: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    rng = rng or random.Random()
    paths = resolve_pool_paths(env)
    path = pool_path if pool_path is not None else paths.dev
    if path is None:
        raise ProbePoolError(
            "no dev probe pool configured; set SCE_DEV_PROBE_POOL to a path "
            "outside the git repo"
        )
    rows = load_dev_pool(path, env=env if env is not None else os.environ)
    if not rows:
        raise ProbePoolError("dev probe pool is empty")
    return rng.choice(rows)


def aggregate_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts only. Ids, families, and text are not copied into the result."""
    probe_count = 0
    families: set[str] = set()
    rejected_burned = 0
    duplicate_ids = 0
    rejected_rows = 0
    seen: set[str] = set()
    synthetic_flags: list[bool] = []
    for row in records:
        if not isinstance(row, dict):
            rejected_rows += 1
            continue
        probe_id = row.get("probe_id")
        if type(probe_id) is not str or not probe_id:
            rejected_rows += 1
            continue
        if probe_id in BURNED_PROBE_IDS:
            rejected_burned += 1
            continue
        if probe_id in seen:
            duplicate_ids += 1
            continue
        seen.add(probe_id)
        probe_count += 1
        family = row.get("family")
        if type(family) is str and family:
            families.add(family)
        synthetic_flags.append(row.get("synthetic") is True)
    return {
        "probe_count": probe_count,
        "family_count": len(families),
        "rejected_burned_ids": rejected_burned,
        "duplicate_ids": duplicate_ids,
        "rejected_rows": rejected_rows,
        "synthetic": bool(synthetic_flags) and all(synthetic_flags),
        "held_out_validation": False,
        "blocks_refund": False,
    }


def parse_aggregate_stdout(stdout: str) -> dict[str, Any]:
    """Accept the aggregate object and nothing else. The error omits stdout."""
    try:
        data = json.loads(stdout)
    except json.JSONDecodeError:
        raise SealedAccessError("sealed child returned a non-aggregate payload") from None
    if not isinstance(data, dict) or set(data) != AGGREGATE_KEYS:
        raise SealedAccessError("sealed child returned a non-aggregate payload")
    if data["held_out_validation"] is not False or data["blocks_refund"] is not False:
        raise SealedAccessError("sealed child returned a non-aggregate payload")
    if type(data["synthetic"]) is not bool:
        raise SealedAccessError("sealed child returned a non-aggregate payload")
    for key in _COUNT_KEYS:
        if type(data[key]) is not int or data[key] < 0:
            raise SealedAccessError("sealed child returned a non-aggregate payload")
    return data


def collect_sealed_aggregate(
    *,
    runner: Callable[[], tuple[int, str]] | None = None,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run the sealed child, or ``runner``, and return counts."""
    code, stdout = _run_child(env) if runner is None else runner()
    data = parse_aggregate_stdout(stdout)
    if code != 0:
        raise ProbePoolError("sealed pool rejected")
    return data


def _run_child(env: Mapping[str, str] | None) -> tuple[int, str]:
    run_env = dict(os.environ if env is None else env)
    src = str(Path(__file__).resolve().parents[1])
    current = run_env.get("PYTHONPATH", "")
    parts = [part for part in current.split(os.pathsep) if part]
    if src not in parts:
        parts.insert(0, src)
    run_env["PYTHONPATH"] = os.pathsep.join(parts)
    proc = subprocess.run(
        [sys.executable, "-m", "probe_runtime.sealed_aggregate"],
        capture_output=True,
        text=True,
        env=run_env,
    )
    return proc.returncode, proc.stdout
