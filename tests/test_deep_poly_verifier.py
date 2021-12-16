import numpy as np
import torch
from torch import nn

from deep_poly_verifier import DeepPolyVerifier

def test_simple_liner_relu():
    # Setup
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])

    l2 = nn.Linear(2, 2)
    l2.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l2.bias[:] = torch.Tensor([-0.5, 0])

    l3 = nn.Linear(2, 2)
    l3.weight[:] = torch.Tensor([[-1, 1], [0, 1]])
    l3.bias[:] = torch.Tensor([3, 0])

    net = nn.Sequential(
        l1, nn.ReLU(),
        l2, nn.ReLU(),
        l3)

    input = torch.Tensor([0, 0])
    eps = 1
    verifier = DeepPolyVerifier(net)

    # Execution
    verified = verifier.verify(input, eps, 0, verbose = True)

    # Validation
    assert verified == True

    dpoly = verifier.dpolys[-1]
    assert np.all(np.isclose(dpoly.box.l, [0.5, 0]))
    assert np.all(np.isclose(dpoly.box.u, [5, 2]))


if __name__ == "__main__":
    test_simple_liner_relu()