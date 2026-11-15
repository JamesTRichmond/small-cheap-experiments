"""Equivalent-wording variants of a dev-pool probe.

Each variant is produced from the original text by one deterministic
transform. Transforms are not chained. The result is a tuple of

    (variant_id, base_probe_id, transform_name, text)

Claim-stability scoring belongs to the scoring lane. This module does not
compare claims, does not emit an unstable signal, and does not read the
sealed pool or the public holdout.

The paraphrase table is the whole paraphrase. Phrases that are not in the
table are left verbatim and, when nothing in the table matches, wrapped in
a frame that says the question is unchanged. No model is called.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, NamedTuple

# Public ids. Variants are not minted from them.
BURNED_PROBE_IDS = frozenset({"goal-001", "inc-001", "rev-001", "rev-002"})

SEALED_POOL_ENV = "SCE_SEALED_PROBE_POOL"

# Meaning-preserving replacements, applied in this order to the original text.
PARAPHRASE_TABLE = (
    (
        "Should this refund proceed?",
        "Is this refund something you would still execute?",
    ),
    ("Reply JSON only", "Respond with JSON only"),
    ("JSON:", "Reply with JSON:"),
)

_PARAPHRASE_FRAME = "Unchanged question follows.\n"

_DISTRACTOR = (
    "Irrelevant note, not part of the question: the hallway clock is round."
)

_ONES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)


class ProbePoolPathError(RuntimeError):
    """The dev-pool path is inside the repo, sealed, or unreadable."""


class ProbeVariant(NamedTuple):
    variant_id: str
    base_probe_id: str
    transform_name: str
    text: str


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _sealed_root(sealed: Path | None, env: Mapping[str, str]) -> Path | None:
    if sealed is not None:
        return Path(sealed)
    raw = env.get(SEALED_POOL_ENV) or None
    return Path(raw) if raw else None


def _amount_words(amount: int) -> str:
    """Words for an integer from 0 through 9999. Hyphens are not used."""
    if amount < 20:
        return _ONES[amount]
    if amount < 100:
        tens, ones = divmod(amount, 10)
        return _TENS[tens] if ones == 0 else f"{_TENS[tens]} {_ONES[ones]}"
    if amount < 1000:
        hundreds, rest = divmod(amount, 100)
        head = f"{_ONES[hundreds]} hundred"
        return head if rest == 0 else f"{head} {_amount_words(rest)}"
    thousands, rest = divmod(amount, 1000)
    head = f"{_ONES[thousands]} thousand"
    return head if rest == 0 else f"{head} {_amount_words(rest)}"


def _paraphrase(text: str) -> str:
    rewritten = text
    for old, new in PARAPHRASE_TABLE:
        rewritten = rewritten.replace(old, new)
    if rewritten == text:
        return _PARAPHRASE_FRAME + text
    return rewritten


def _reorder_fields(text: str, fields: object) -> str | None:
    if not isinstance(fields, list) or len(fields) < 2:
        return None
    if any(type(field) is not str or not field for field in fields):
        return None
    original = ", ".join(fields)
    if original not in text:
        return None
    reversed_fields = ", ".join(reversed(fields))
    return text.replace(original, reversed_fields, 1)


def _amount_variant(text: str, amount: object) -> str | None:
    if type(amount) is not int or isinstance(amount, bool):
        return None
    if amount < 0 or amount > 9999:
        return None
    digits = str(amount)
    if digits not in text:
        return None
    return text.replace(digits, _amount_words(amount), 1)


def _distractor(text: str) -> str:
    return text + "\n" + _DISTRACTOR


def equivalent_variants(probe: Mapping[str, Any]) -> tuple[ProbeVariant, ...]:
    """Wording variants of one probe. Does not score the probe."""
    probe_id = probe.get("probe_id")
    text = probe.get("text")
    if type(probe_id) is not str or not probe_id.strip():
        raise ValueError("probe_id must be a non-empty string")
    if probe_id in BURNED_PROBE_IDS:
        raise ValueError(f"burned probe id reused: {probe_id}")
    if type(text) is not str or not text.strip():
        raise ValueError("probe text must be a non-empty string")

    drafted: list[tuple[str, str]] = [("template_paraphrase", _paraphrase(text))]
    reordered = _reorder_fields(text, probe.get("fields"))
    if reordered is not None and reordered != text:
        drafted.append(("reordered_fields", reordered))
    amount = _amount_variant(text, probe.get("amount_minor"))
    if amount is not None and amount != text:
        drafted.append(("amount_words", amount))
    drafted.append(("distractor", _distractor(text)))

    return tuple(
        ProbeVariant(
            variant_id=f"{probe_id}:{name}",
            base_probe_id=probe_id,
            transform_name=name,
            text=variant,
        )
        for name, variant in drafted
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        raise ProbePoolPathError("dev probe pool could not be read") from None
    except UnicodeDecodeError:
        raise ProbePoolPathError("dev probe pool is not utf-8 text") from None
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.split("\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ProbePoolPathError(
                f"invalid json in dev probe pool at line {line_number}"
            ) from None
        if not isinstance(row, dict):
            raise ProbePoolPathError(f"dev probe pool line {line_number} is not an object")
        rows.append(row)
    return rows


def variants_from_dev_pool(
    path: Path,
    *,
    sealed: Path | None = None,
    env: Mapping[str, str] | None = None,
    repo: Path | None = None,
) -> tuple[ProbeVariant, ...]:
    """Read a dev-pool jsonl outside the repo and return wording variants.

    A path inside the sealed root raises before that path is opened.
    """
    env = os.environ if env is None else env
    repo = (repo or repo_root()).resolve()
    target = Path(path)
    sealed_root = _sealed_root(sealed, env)
    if sealed_root is not None and _within(target, sealed_root):
        raise ProbePoolPathError("variant generation cannot read the sealed pool")
    if _within(target, repo):
        raise ProbePoolPathError(
            "variant generation does not read probes inside the git repo"
        )
    if target.is_dir():
        files = sorted(item for item in target.glob("*.jsonl") if item.is_file())
    elif target.is_file():
        files = [target]
    else:
        raise ProbePoolPathError("dev probe pool path does not exist")
    variants: list[ProbeVariant] = []
    for file in files:
        for row in _read_jsonl(file):
            variants.extend(equivalent_variants(row))
    return tuple(variants)
