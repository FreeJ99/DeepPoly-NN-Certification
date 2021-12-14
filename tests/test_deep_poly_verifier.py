import numpy as np
import torch
from torch import nn

from deep_poly_verifier import DeepPolyVerifier

def test_simple_liner_relu():
    # Setup
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])
    net = nn.Sequential(l1, nn.ReLU())

    input = torch.Tensor([0, 0])
    eps = 1
    verifier = DeepPolyVerifier(net, input, eps, 0)

    expected_box_l_list = [
        np.array([-1, -1]),
        np.array([-2, -2]),
        np.array([0, 0])
    ]
    expected_box_u_list = [
        np.array([1, 1]),
        np.array([2, 2]),
        np.array([2, 2])
    ]
    expected_l_bias_list = [
        np.array([-1, -1]),
        np.array([0, 0]),
        np.array([0, 0])
    ]
    expected_u_bias_list = [
        np.array([1, 1]),
        np.array([0, 0]),
        np.array([1, 1])
    ]
    expected_l_weights_list = [
        np.array([[0], [0]]),
        np.array([[1, 1], [1, -1]]),
        np.array([[0, 0], [0, 0]]),
    ]
    expected_u_weights_list = [
        np.array([[0], [0]]),
        np.array([[1, 1], [1, -1]]),
        np.array([[0.5, 0], [0, 0.5]]),
    ]

    # Execution
    verified = verifier.verify()

    # Validation
    assert verified == False

    for (dpoly, 
            expected_box_l, expected_box_u,
            expected_l_bias, expected_u_bias, 
            expected_l_weights, expected_u_weights) in zip(
            verifier.dpolys, 
            expected_box_l_list, expected_box_u_list,
            expected_l_bias_list, expected_u_bias_list,
            expected_l_weights_list, expected_u_weights_list):
        assert np.isclose(dpoly.box.l, expected_box_l).all()
        assert np.isclose(dpoly.box.u, expected_box_u).all()
        assert np.isclose(dpoly.l_bias, expected_l_bias).all()
        assert np.isclose(dpoly.u_bias, expected_u_bias).all()
        assert np.isclose(dpoly.l_weights, expected_l_weights).all()
        assert np.isclose(dpoly.u_weights, expected_u_weights).all()


if __name__ == "__main__":
    test_simple_liner_relu()