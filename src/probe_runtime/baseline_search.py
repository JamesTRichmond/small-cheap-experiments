"""Search a finite family of metadata-only allow rules.

The toy rule in compare_batch is one rule. This module enumerates a
declared family that never reads the claim, scores each rule with
compare_with_allows, and keeps the rule that does best for the rule.
Equal probe and rule performance is STOP, so a tie goes to the rule.
Two rules with the same score keep the one enumerated first.

The winner is fit on the same batch. That favors the rule and is
conservative against the probe. leave_one_out_allows refits without each
row and is a diagnostic, not the headline verdict.

Atoms are tool name, authorized, currency USD, and amount <= a threshold.
Thresholds are a fixed list plus each non-negative amount in the batch,
sorted. Rules are single atoms and conjunctions of two distinct atoms,
in that order, after an always-allow rule.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probe_runtime.head_to_head import (
    VERDICT_EXPERIMENTAL,
    VERDICT_INCONCLUSIVE,
    VERDICT_STOP,
    Comparison,
    compare_with_allows,
)

METADATA_KEYS = ("tool_name", "authorized", "amount_minor", "currency")
FIXED_AMOUNT_THRESHOLDS = (0, 1, 50, 100, 1000, 10000, 50000)
FIXED_CURRENCY = "USD"

_VERDICT_RANK = {
    VERDICT_STOP: 2,
    VERDICT_INCONCLUSIVE: 1,
    VERDICT_EXPERIMENTAL: 0,
}

FIT_NOTE = (
    "The best rule is fit on the same batch. That favors the rule and is "
    "conservative against the probe. It is not a claim about a new batch."
)


@dataclass(frozen=True)
class Atom:
    """One metadata test. value is a tool name, a bool, or an amount ceiling."""

    name: str
    op: str
    value: object


@dataclass(frozen=True)
class Rule:
    """A conjunction of zero, one, or two atoms. Zero atoms always allows."""

    rule_id: str
    atoms: tuple[Atom, ...]

    def allows(self, case: Mapping[str, object]) -> bool:
        meta = {key: case.get(key) for key in METADATA_KEYS}
        return all(_atom_holds(atom, meta) for atom in self.atoms)


@dataclass(frozen=True)
class BaselineSearch:
    """In-sample winner. leave_one_out_allows is None when the batch is shorter than 2."""

    rule_id: str
    rule: Rule
    comparison: Comparison
    family_size: int
    fit_on_same_batch: bool
    overfit_favors_rule: bool
    note: str
    leave_one_out_allows: tuple[bool, ...] | None


def _atom_holds(atom: Atom, meta: Mapping[str, object]) -> bool:
    if atom.op == "tool_eq":
        return meta.get("tool_name") == atom.value
    if atom.op == "tool_ne":
        return meta.get("tool_name") != atom.value
    if atom.op == "auth_is":
        if atom.value is True:
            return meta.get("authorized") is True
        return meta.get("authorized") is not True
    if atom.op == "currency_eq":
        return meta.get("currency") == atom.value
    if atom.op == "currency_ne":
        return meta.get("currency") != atom.value
    if atom.op == "amount_le":
        amount = meta.get("amount_minor")
        return type(amount) is int and amount <= atom.value
    return False


def _amount_thresholds(cases: Sequence[Mapping[str, object]]) -> tuple[int, ...]:
    found = set(FIXED_AMOUNT_THRESHOLDS)
    for case in cases:
        amount = case.get("amount_minor")
        if type(amount) is int and amount >= 0:
            found.add(amount)
    return tuple(sorted(found))


def iter_atoms(cases: Sequence[Mapping[str, object]]) -> tuple[Atom, ...]:
    """Declared atoms, in a fixed order, plus batch amount ceilings sorted."""
    atoms = [
        Atom("tool_eq_refund", "tool_eq", "refund.create"),
        Atom("tool_ne_refund", "tool_ne", "refund.create"),
        Atom("authorized", "auth_is", True),
        Atom("not_authorized", "auth_is", False),
        Atom("currency_eq_usd", "currency_eq", FIXED_CURRENCY),
        Atom("currency_ne_usd", "currency_ne", FIXED_CURRENCY),
    ]
    for amount in _amount_thresholds(cases):
        atoms.append(Atom(f"amount_le_{amount}", "amount_le", amount))
    return tuple(atoms)


def iter_rules(cases: Sequence[Mapping[str, object]]) -> tuple[Rule, ...]:
    """Always-allow, then each atom, then each unordered pair. Deterministic."""
    atoms = iter_atoms(cases)
    rules = [Rule("always_allow", ())]
    for atom in atoms:
        rules.append(Rule(atom.name, (atom,)))
    for index, left in enumerate(atoms):
        for right in atoms[index + 1 :]:
            rules.append(Rule(f"{left.name}&{right.name}", (left, right)))
    return tuple(rules)


def _rule_strength(comparison: Comparison) -> tuple[int, int, int]:
    """Higher is better for the rule. Missing counts sort below a real zero."""
    tp = comparison.tp_baseline if comparison.tp_baseline is not None else -1
    fp = comparison.fp_baseline if comparison.fp_baseline is not None else 10**9
    return (_VERDICT_RANK[comparison.verdict], tp, -fp)


def search_strongest_baseline(
    cases: Sequence[Mapping[str, object]],
    *,
    leave_one_out: bool = True,
) -> BaselineSearch:
    """Pick the metadata rule that most favors the rule on this same batch."""
    rules = iter_rules(cases)
    best_rule = rules[0]
    best_comparison = compare_with_allows(cases, best_rule.allows)
    best_key = (_rule_strength(best_comparison), 0)
    for index, rule in enumerate(rules[1:], start=1):
        comparison = compare_with_allows(cases, rule.allows)
        key = (_rule_strength(comparison), -index)
        if key > best_key:
            best_key = key
            best_rule = rule
            best_comparison = comparison
    loo: tuple[bool, ...] | None
    if leave_one_out and len(cases) >= 2:
        loo = tuple(leave_one_out_allows(cases))
    else:
        loo = None
    return BaselineSearch(
        rule_id=best_rule.rule_id,
        rule=best_rule,
        comparison=best_comparison,
        family_size=len(rules),
        fit_on_same_batch=True,
        overfit_favors_rule=True,
        note=FIT_NOTE,
        leave_one_out_allows=loo,
    )


def leave_one_out_allows(cases: Sequence[Mapping[str, object]]) -> list[bool]:
    """Refit on every batch that drops one row, then allow or deny that row.

    Diagnostic only. The headline verdict is the in-sample search.
    """
    decided: list[bool] = []
    for index in range(len(cases)):
        rest = list(cases[:index]) + list(cases[index + 1 :])
        fitted = search_strongest_baseline(rest, leave_one_out=False)
        decided.append(fitted.rule.allows(cases[index]))
    return decided
