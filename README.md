# sce-probe-runtime

[![CI](https://github.com/JamesTRichmond/small-cheap-experiments/actions/workflows/ci.yml/badge.svg)](https://github.com/JamesTRichmond/small-cheap-experiments/actions/workflows/ci.yml)

<!-- portfolio summary -->
> **Portfolio summary.** Runtime telemetry for AI agents that can take irreversible actions. Before a model issues a refund, a silent held-out probe checks whether what it *claims* matches what it *does*, and every decision is sealed into a hash-chained, replayable audit record.
>
> **Stack:** Python · JSON Schema · pytest
>
> **Why it matters:** agent evaluations usually grade the transcript. This grades the action, at the moment it happens, with evidence a risk team can replay.

Thin MVP: silent held-out micro-AMBUSH on **refund** → score claim↔next tool → hash-chained three-frame SOC replay.

**Not in v0:** Diet tribunal UX, Boar continuous hunt, dashboards-as-product, second tool surfaces.

Canon: room-ratified `PROBE_RUNTIME_BLUEPRINT.md` (Hassabis ratification + Hinton blueprint + Thiel filter).

## Non-substitutables
1. Silent held-out probe every N refunds (pilot N=1)
2. Score = claim ↔ refund tool mismatch (not prose)
3. Three-frame replay Risk cannot rip out (`sce.replay.v1`)

## Go criterion after skeleton
One Risk conversation naming the existing chargeback/dispute P&L line before feature work past the spike.
