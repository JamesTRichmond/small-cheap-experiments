"""Child entry point for the sealed probe pool.

``python -m probe_runtime.sealed_aggregate`` reads ``SCE_SEALED_PROBE_POOL``
and writes one JSON object of counts to stdout. Stderr is a fixed sentence
when the pool is missing or rejected. Probe text, probe ids, family names,
and paths are not written.

``held_out_validation`` is false on every aggregate. A count from this process
is not a measurement, including when the file was a synthetic fixture.
``blocks_refund`` is false. This process does not decide a refund.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Mapping

from probe_runtime.probe_pool import (
    ProbePoolError,
    aggregate_records,
    read_probe_records,
    repo_root,
    resolve_pool_paths,
)


def _inside_repo(path: Path) -> bool:
    try:
        path.resolve().relative_to(repo_root().resolve())
    except ValueError:
        return False
    return True


def main(env: Mapping[str, str] | None = None) -> int:
    try:
        sealed = resolve_pool_paths(env).sealed
    except ProbePoolError:
        sys.stderr.write("sealed pool rejected\n")
        return 2
    if sealed is None:
        sys.stderr.write("sealed probe pool is not configured\n")
        return 2
    if _inside_repo(sealed):
        sys.stderr.write("sealed pool rejected\n")
        return 2
    try:
        records = read_probe_records(sealed)
    except ProbePoolError:
        sys.stderr.write("sealed pool rejected\n")
        return 2
    aggregate = aggregate_records(records)
    sys.stdout.write(json.dumps(aggregate, sort_keys=True) + "\n")
    if (
        aggregate["rejected_burned_ids"]
        or aggregate["duplicate_ids"]
        or aggregate["rejected_rows"]
    ):
        sys.stderr.write("sealed pool rejected\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
