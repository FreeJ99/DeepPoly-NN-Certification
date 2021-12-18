import numpy as np
import torch
from torch import nn

from deep_poly_verifier import DeepPolyVerifier
from networks import FullyConnected, Normalization

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
    dpoly = verifier.dpolys[-1]

    # Validation
    assert verified == True
    assert np.all(np.isclose(dpoly.box.l, [0.5, 0]))
    assert np.all(np.isclose(dpoly.box.u, [5, 2]))

def test_simple_fc_net():
    net = FullyConnected('cpu', 2, [5, 2])
    net.layers[0].mean = torch.FloatTensor([1]).view((1, 1, 1, 1)).to('cpu')
    net.layers[0].sigma = torch.FloatTensor([0.5]).view((1, 1, 1, 1)).to('cpu')
    net.layers[2].weight[:] = torch.Tensor([
        [1, 0, 1, 0],
        [1, -1, 1, -1],
        [0, 0, 0, 1],
        [-1, 1, 1, 0],
        [1, -1, 1, 1]
    ])
    net.layers[2].bias[:] = torch.Tensor([0, -1, 0, 1, 1])
    net.layers[4].weight[:] = torch.Tensor([
        [1, -1, 0, -1, 1],
        [1, 0, -1, 0, 1]
    ])
    net.layers[4].bias[:] = torch.Tensor([1, -1])

    input = torch.FloatTensor([[1, -1], [2, 3]]).view(1, 1, 2, 2)
    verifier = DeepPolyVerifier(net)

    # Test 1
    eps = 1
    verified = verifier.verify(input, eps, 0, verbose = False)
    dpoly = verifier.dpolys[-1]

    assert verified == False
    assert np.all(np.isclose(dpoly.box.l, [-1.5833333, 0]))
    assert np.all(np.isclose(dpoly.box.u, [26,18]))

    # Test 2
    eps = 0.1
    verified = verifier.verify(input, eps, 0, verbose = False)
    dpoly = verifier.dpolys[-1]

    assert verified == True
    assert np.all(np.isclose(dpoly.box.l, [12.2, 7]))
    assert np.all(np.isclose(dpoly.box.u, [13.8, 9]))

if __name__ == "__main__":
    test_simple_liner_relu()