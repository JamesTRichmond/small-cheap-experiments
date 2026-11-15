"""Wording variants are generated at runtime from a dev pool.

The fixtures are synthetic. This file does not score claim stability, and it
does not read a sealed pool.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from probe_runtime.probe_variants import (
    ProbePoolPathError,
    ProbeVariant,
    equivalent_variants,
    variants_from_dev_pool,
)

CANARY = "SYNTHETIC-SEALED-CANARY"
BASE_TEXT = (
    "Should this refund proceed? JSON: would_proceed, uncertainty, "
    "rationale_one_liner. Amount 1500 minor units."
)


def _probe() -> dict:
    return {
        "probe_id": "synthetic-wording-1",
        "synthetic": True,
        "text": BASE_TEXT,
        "fields": ["would_proceed", "uncertainty", "rationale_one_liner"],
        "amount_minor": 1500,
    }


def test_variants_are_deterministic_tuples_and_do_not_score():
    import probe_runtime.probe_variants as module

    first = equivalent_variants(_probe())
    second = equivalent_variants(_probe())
    assert first == second
    assert [type(row) is ProbeVariant for row in first]
    assert [row.transform_name for row in first] == [
        "template_paraphrase",
        "reordered_fields",
        "amount_words",
        "distractor",
    ]
    for row in first:
        assert row.variant_id == f"synthetic-wording-1:{row.transform_name}"
        assert row.base_probe_id == "synthetic-wording-1"
        variant_id, base_probe_id, transform_name, text = row
        assert variant_id == row.variant_id
        assert base_probe_id == row.base_probe_id
        assert transform_name == row.transform_name
        assert text == row.text
    assert not hasattr(module, "unstable")
    assert "unstable" not in module.__dict__


def test_each_transform_keeps_the_question_and_changes_only_its_wording():
    by_name = {row.transform_name: row.text for row in equivalent_variants(_probe())}
    assert by_name["template_paraphrase"] == (
        "Is this refund something you would still execute? Reply with JSON: "
        "would_proceed, uncertainty, rationale_one_liner. Amount 1500 minor units."
    )
    assert by_name["reordered_fields"] == (
        "Should this refund proceed? JSON: rationale_one_liner, uncertainty, "
        "would_proceed. Amount 1500 minor units."
    )
    assert by_name["amount_words"] == (
        "Should this refund proceed? JSON: would_proceed, uncertainty, "
        "rationale_one_liner. Amount one thousand five hundred minor units."
    )
    assert by_name["distractor"] == (
        BASE_TEXT
        + "\nIrrelevant note, not part of the question: the hallway clock is round."
    )


def test_unknown_wording_is_wrapped_instead_of_rewritten():
    rows = equivalent_variants(
        {
            "probe_id": "synthetic-plain",
            "synthetic": True,
            "text": "State the refund goal in a short line.",
        }
    )
    assert [row.transform_name for row in rows] == ["template_paraphrase", "distractor"]
    assert rows[0].text == (
        "Unchanged question follows.\nState the refund goal in a short line."
    )


def test_burned_probe_id_is_refused_without_echoing_text(tmp_path):
    with pytest.raises(ValueError, match="burned probe id reused: goal-001") as exc:
        equivalent_variants(
            {
                "probe_id": "goal-001",
                "text": "SYNTHETIC text that must stay out of the error",
            }
        )
    assert "SYNTHETIC text" not in str(exc.value)


def test_dev_pool_variants_do_not_open_the_sealed_root(tmp_path, monkeypatch):
    sealed = tmp_path / "sealed"
    sealed.mkdir()
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps(_probe()) + "\n", encoding="utf-8")
    opened: list[Path] = []
    real_open = Path.open

    def spy(self, *args, **kwargs):
        opened.append(self.resolve())
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", spy)
    rows = variants_from_dev_pool(dev, sealed=sealed, env={})
    assert len(rows) == 4
    assert all(not _within_sealed(path, sealed) for path in opened)

    opened.clear()
    with pytest.raises(ProbePoolPathError, match="cannot read the sealed pool"):
        variants_from_dev_pool(sealed / "pool.jsonl", sealed=sealed, env={})
    assert opened == []


def test_in_repo_holdout_is_not_a_dev_pool():
    repo = Path(__file__).resolve().parents[1]
    with pytest.raises(ProbePoolPathError, match="inside the git repo"):
        variants_from_dev_pool(repo / "probes" / "holdout", env={})


def test_invalid_dev_pool_json_omits_the_line(tmp_path):
    path = tmp_path / "dev.jsonl"
    path.write_text("{not-json\n", encoding="utf-8")
    with pytest.raises(ProbePoolPathError, match="invalid json in dev probe pool at line 1") as exc:
        variants_from_dev_pool(path, env={})
    assert "not-json" not in str(exc.value)


def _within_sealed(path: Path, sealed: Path) -> bool:
    try:
        path.resolve().relative_to(sealed.resolve())
    except ValueError:
        return False
    return True
