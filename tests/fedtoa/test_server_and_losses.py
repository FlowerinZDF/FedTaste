import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from methods.fedtoa.losses import (  # noqa: E402
    fedtoa_total_loss,
    masked_topology_loss,
    spectral_consistency_loss,
)
from methods.fedtoa.server_ops import (  # noqa: E402
    aggregate_topologies_mean,
    aggregate_topologies_var,
    build_confidence_mask,
    build_global_blueprint,
)


def test_server_aggregation_and_blueprint_building():
    topologies = torch.tensor(
        [
            [[0.0, 2.0, 0.0], [2.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 3.0], [0.0, 3.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    mean = aggregate_topologies_mean(topologies)
    var = aggregate_topologies_var(topologies)

    conf = build_confidence_mask(mean, var, topk_edges=1)
    assert conf.sum().item() == 2  # one undirected edge

    spectral_list = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32)
    class_masks = torch.tensor([[1, 1, 0], [1, 0, 1]], dtype=torch.bool)
    blueprint = build_global_blueprint(mean, conf, spectral_list, class_masks)

    assert blueprint.topology_mean.shape == (3, 3)
    assert torch.equal(blueprint.topology_mask, conf)
    assert torch.allclose(blueprint.spectral_global, torch.tensor([0.2, 0.3]))
    assert torch.equal(blueprint.active_classes, torch.tensor([True, True, True]))


def test_masked_topology_loss_support_mask_and_zero_case():
    local = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    global_ = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    edge_mask = torch.tensor([[0, 1], [1, 0]], dtype=torch.bool)

    loss = masked_topology_loss(
        local_topology=local,
        global_topology=global_,
        edge_mask=edge_mask,
        class_support_mask=torch.tensor([True, True]),
    )
    assert torch.isclose(loss, torch.tensor(1.0))

    zero_loss = masked_topology_loss(
        local_topology=local,
        global_topology=global_,
        edge_mask=edge_mask,
        class_support_mask=torch.tensor([True, False]),
    )
    assert torch.isclose(zero_loss, torch.tensor(0.0))


def test_spectral_and_total_loss():
    spec = spectral_consistency_loss(
        torch.tensor([1.0, 2.0]),
        torch.tensor([2.0, 2.0]),
    )
    assert torch.isclose(spec, torch.tensor(0.5))

    total = fedtoa_total_loss(
        task_loss=torch.tensor(1.0),
        topo_loss=torch.tensor(2.0),
        spec_loss=torch.tensor(3.0),
        lip_loss=torch.tensor(4.0),
        beta=0.1,
        gamma=0.2,
        eta=0.3,
    )
    assert torch.isclose(total, torch.tensor(3.0))
