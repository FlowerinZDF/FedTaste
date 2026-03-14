# AGENTS.md

Project goal:
Implement FedToA on top of FedCola with minimal invasive changes.

Rules:
- Do not rewrite main.py unless necessary.
- Reuse existing data loaders, scripts, and federated training flow.
- Add a new algorithm named `fedtoa`.
- Prefer adding new files under `src/methods/fedtoa/`, `src/client/`, `src/server/`.
- Do not remove existing CreamFL code.
- Add tests for pure functions before wiring them into FL flow.
- Make small, reviewable patches.
- After each task, summarize modified files and validation steps.
