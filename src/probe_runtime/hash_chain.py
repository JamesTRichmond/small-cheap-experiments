"""Append-only hash chain for FrameTriplet records."""

from __future__ import annotations

import hashlib
import json
from typing import Any


GENESIS = "genesis"


def canonical_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record_hash(payload_without_integrity: dict[str, Any], prev_hash: str) -> str:
    blob = canonical_json(payload_without_integrity) + "||" + prev_hash
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def attach_integrity(record: dict[str, Any], prev_hash: str = GENESIS) -> dict[str, Any]:
    body = {k: v for k, v in record.items() if k != "integrity"}
    rh = record_hash(body, prev_hash)
    out = dict(body)
    out["integrity"] = {"prev_hash": prev_hash, "record_hash": rh}
    return out


def verify_chain(records: list[dict[str, Any]]) -> bool:
    prev = GENESIS
    for rec in records:
        integ = rec.get("integrity") or {}
        if integ.get("prev_hash") != prev:
            return False
        body = {k: v for k, v in rec.items() if k != "integrity"}
        if record_hash(body, prev) != integ.get("record_hash"):
            return False
        prev = integ["record_hash"]
    return True
