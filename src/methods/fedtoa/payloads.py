"""FedToA payload dataclasses.

These payloads are intentionally lightweight containers used by client/server
orchestration layers and pure-method utilities.
"""

from dataclasses import dataclass

import torch


@dataclass
class TeacherTopologyPayload:
    """Teacher upload payload for FedToA.

    Attributes:
        client_id: Teacher client identifier.
        class_ids: Local class ids present on teacher, shape ``[C_local]``.
        topology: Class-level topology aligned to global class index space,
            shape ``[C_global, C_global]``.
        spectral: Teacher spectral signature, shape ``[K]``.
        support_mask: Global class support mask for this teacher,
            shape ``[C_global]`` with bool/0-1 entries.
        num_samples: Number of local samples used to build summaries.
    """

    client_id: int
    class_ids: torch.Tensor
    topology: torch.Tensor
    spectral: torch.Tensor
    support_mask: torch.Tensor
    num_samples: int


@dataclass
class GlobalTopologyBlueprint:
    """Global topology blueprint broadcast by FedToA server.

    Attributes:
        topology_mean: Aggregated global class topology, shape
            ``[C_global, C_global]``.
        topology_mask: Confidence-filtered sparse edge mask, shape
            ``[C_global, C_global]`` with bool/0-1 entries.
        spectral_global: Aggregated spectral signature, shape ``[K]``.
        active_classes: Class activity mask inferred from teacher supports,
            shape ``[C_global]`` with bool/0-1 entries.
    """

    topology_mean: torch.Tensor
    topology_mask: torch.Tensor
    spectral_global: torch.Tensor
    active_classes: torch.Tensor


@dataclass
class FedToAConfig:
    """FedToA hyperparameter and switch configuration."""

    tau: float
    eig_k: int
    topk_edges: int
    beta_topo: float
    gamma_spec: float
    eta_lip: float
    prompt_len: int
    diagonal_eps: float
