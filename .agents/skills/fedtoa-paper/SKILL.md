---
name: fedtoa-paper
description: Use this skill for any FedToA implementation, review, debugging, or experiment scripting task in this repository.
---

Always read:
- docs/fedtoa_spec.md
- docs/fedtoa_plan.md
- docs/fedtoa_api.md

Treat docs/fedtoa_spec.md as the source of truth.

Workflow:
1. Understand the repository scope before editing code.
2. Prefer minimal invasive changes.
3. Add new FedToA logic in dedicated files.
4. Implement pure method modules before FL wiring.
5. Add tests for pure functions before integration.
6. Keep diffs reviewable and scoped.
7. Report modified files, validation steps, and open risks after each task.

FedToA-specific rules:
- Topology is class-level, not sample-level.
- Teacher clients upload topology and spectral summaries.
- Server aggregates topology with variance-aware filtering.
- Student clients update prompt parameters only by default.
- Do not add external public-data distillation into FedToA.
- Keep use_masp, use_topo, use_spec, and use_lip configurable.

Review checklist:
- tensor shapes consistent
- support masks handled correctly
- eigendecomposition numerically stabilized
- no accidental backbone full-finetuning
- no unnecessary changes outside FedToA scope
