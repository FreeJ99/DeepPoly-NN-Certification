import math

import torch
from torch import nn
import numpy as np

from networks import Normalization, FullyConnected
from box import BoxVerifier

def test_fc_relu_2layers_verify():
    # Setup
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])
    relu1 = nn.ReLU()
    l2 = nn.Linear(2, 2)
    l2.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l2.bias[:] = torch.Tensor([0.5, -0.5])
    net = nn.Sequential(l1, relu1, l2)

    input = torch.Tensor([0.15, 0.25])
    eps = 0.15
    verifier = BoxVerifier(net)
    
    expected_lower_bounds = [
        np.array([0, 0.1]),
        np.array([0.1, -0.4]),
        np.array([0.1, 0]),
        np.array([0.6, -0.6])
    ]
    expected_upper_bounds = [
        np.array([0.3, 0.4]),
        np.array([0.7, 0.2]),
        np.array([0.7, 0.2]),
        np.array([1.4, 0.2])
    ]

    # Execution
    verified = verifier.verify(input, eps, 0)

    # Validation
    assert verified == True

    for box, expected_l, expected_u in zip(verifier.boxes, 
            expected_lower_bounds, expected_upper_bounds):
        assert np.isclose(box.l, expected_l).all()
        assert np.isclose(box.u, expected_u).all()

    print()


def test_fc_relu_2layers_fail():
    # Setup
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])
    relu1 = nn.ReLU()
    l2 = nn.Linear(2, 2)
    l2.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l2.bias[:] = torch.Tensor([0.5, -0.5])
    net = nn.Sequential(l1, relu1, l2)

    input = torch.Tensor([0.3, 0.4])
    eps = 0.3
    verifier = BoxVerifier(net)
    
    expected_lower_bounds = [
        np.array([0, 0.1]),
        np.array([0.1, -0.7]),
        np.array([0.1, 0]),
        np.array([0.6, -0.9])
    ] 
    expected_upper_bounds = [
        np.array([0.6, 0.7]),
        np.array([1.3, 0.5]),
        np.array([1.3, 0.5]),
        np.array([2.3, 0.8])
    ]

    # Execution
    verified = verifier.verify(input, eps, 0)

    # Validation
    assert verified == False

    for box, expected_l, expected_u in zip(verifier.boxes, 
            expected_lower_bounds, expected_upper_bounds):
        assert np.isclose(box.l, expected_l).all()
        assert np.isclose(box.u, expected_u).all()

    print()

if __name__ == "__main__":
    test_fc_relu_2layers_verify()
    test_fc_relu_2layers_fail()