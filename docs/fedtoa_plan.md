# FedToA Implementation Plan

## Milestone 1: Pure method modules

Add these files:
- src/methods/fedtoa/payloads.py
- src/methods/fedtoa/topology.py
- src/methods/fedtoa/server_ops.py
- src/methods/fedtoa/losses.py

Requirements:
- pure PyTorch implementation
- no dependency on federated orchestration
- clear docstrings with tensor shapes
- numerical stabilization for graph and spectral operations

Validation:
- topology unit tests pass
- spectral computations are numerically stable
- no changes to main.py, dataloaders, or current client/server files

## Milestone 2: Prompt and local student adaptation

Add:
- src/methods/fedtoa/prompt.py
- prompt-related logic in src/client/fedtoaclient.py

Requirements:
- frozen backbone by default
- prompt-only training by default
- topology loss, spectral consistency loss, and Lipschitz regularization supported
- support class masks for partial class coverage

Validation:
- only prompt parameters are trainable
- local backward pass succeeds
- shape checks pass

## Milestone 3: Client/server wiring

Add:
- src/client/fedtoaclient.py
- src/server/fedtoaserver.py

Edit minimally:
- algorithm registration so --algorithm fedtoa works

Requirements:
- reuse existing FL loop where possible
- keep existing algorithms intact
- teacher uploads topology payloads
- server aggregates global blueprint
- student consumes blueprint for local training

Validation:
- small smoke test with a toy setup
- no breakage to existing algorithms
- import/config path works

## Milestone 4: Experiment scripts

Add:
- scripts/fedtoa/flickr_smoke.sh
- scripts/fedtoa/flickr_default.sh
- scripts/fedtoa/flickr_alpha01.sh
- scripts/fedtoa/flickr_low_participation.sh
- scripts/fedtoa/flickr_ablation.sh
- scripts/fedtoa/coco_default.sh

Requirements:
- defaults follow docs/fedtoa_spec.md
- output/logging args included
- ablation switches exposed

Validation:
- scripts launch without argument/config errors
- smoke script is runnable

## Milestone 5: Review and cleanup

Tasks:
- review all FedToA-related diffs
- remove unnecessary modifications
- confirm no accidental full-finetuning
- check numerical stability around eigendecomposition
- verify ablation toggles
- verify naming consistency

Validation:
- all tests pass
- smoke script passes
- file scope remains minimal
