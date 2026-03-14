import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from methods.fedtoa.topology import (  # noqa: E402
    build_normalized_laplacian,
    build_topology_matrix,
    compute_class_prototypes,
    fuse_joint_prototypes,
    spectral_signature,
)


def test_compute_class_prototypes_and_support_mask():
    feats = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)

    proto, support = compute_class_prototypes(feats, labels, num_classes=3, normalize=False)

    expected = torch.tensor([[1.5, 0.0], [0.0, 1.5], [0.0, 0.0]], dtype=torch.float32)
    assert torch.allclose(proto, expected)
    assert torch.equal(support, torch.tensor([True, True, False]))


def test_fuse_joint_prototypes_union_support():
    img = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32)
    txt = torch.tensor([[0.0, 1.0], [0.0, 2.0]], dtype=torch.float32)
    m_img = torch.tensor([True, False])
    m_txt = torch.tensor([True, True])

    joint, support = fuse_joint_prototypes(img, txt, m_img, m_txt, normalize=False)

    assert torch.allclose(joint[0], torch.tensor([0.5, 0.5]))
    assert torch.allclose(joint[1], torch.tensor([0.0, 2.0]))
    assert torch.equal(support, torch.tensor([True, True]))


def test_topology_matrix_respects_support_and_zero_diag():
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32)
    support = torch.tensor([True, True, False])

    topo = build_topology_matrix(prototypes, support, tau=0.5, zero_diag=True)

    assert topo.shape == (3, 3)
    assert torch.allclose(torch.diag(topo), torch.zeros(3))
    assert torch.allclose(topo[2], torch.zeros(3))
    assert torch.allclose(topo[:, 2], torch.zeros(3))


def test_laplacian_and_spectral_signature_are_stable():
    topology = torch.tensor(
        [
            [0.0, 1.0, 0.5],
            [1.0, 0.0, 0.2],
            [0.5, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )

    lap = build_normalized_laplacian(topology, eps=1e-4)
    sig = spectral_signature(lap, k=2)

    assert torch.allclose(lap, lap.T, atol=1e-6)
    assert torch.isfinite(lap).all()
    assert sig.shape == (2,)
    assert torch.isfinite(sig).all()
    assert torch.all(sig >= 0)
