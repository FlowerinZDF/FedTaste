# AGENTS.md

Project goal:
Implement FedToA on top of FedCola with minimal invasive changes.

Mandatory reading for any FedToA-related task:
- docs/fedtoa_spec.md
- docs/fedtoa_plan.md
- docs/fedtoa_api.md

Source of truth:
- Treat docs/fedtoa_spec.md as the algorithm source of truth.
- Do not infer FedToA details from memory if they conflict with the spec.

Repository rules:
- Keep changes minimally invasive.
- Prefer adding new files instead of rewriting existing ones.
- Add new FedToA logic under:
  - src/methods/fedtoa/
  - src/client/fedtoaclient.py
  - src/server/fedtoaserver.py
  - scripts/fedtoa/
- Do not remove or rewrite existing FedCola / CreamFL code unless explicitly requested.
- Do not redesign dataloaders unless absolutely necessary.
- Do not rewrite main.py unless it is required to register algorithm=fedtoa.

Implementation rules:
- Plan before coding for any task larger than one file.
- Prefer pure-function modules first, then wiring.
- Add tests for pure functions before integrating into federated flow.
- Use clear docstrings with tensor shapes.
- Add numerical stabilization for eigendecomposition and graph operations.

Validation rules:
- After each coding task, report:
  1. modified files
  2. validation steps
  3. tests run
  4. remaining risks

FedToA-specific constraints:
- Teacher clients upload topology/spectral summaries, not model parameters as the primary FedToA signal.
- Server aggregates topology, variance, and spectral summaries.
- Student clients update prompt parameters only unless explicitly overridden.
- Do not introduce external public-data distillation logic into FedToA.
- Keep component switches configurable:
  - use_masp
  - use_topo
  - use_spec
  - use_lip
