import numpy as np

import torch
from torch import nn

from networks import Normalization
from deep_poly_transform import *
from box import Box


def test_affine_substitute():
    f = np.array([5, 3, -2])
    sub_mat = np.array(
            [[0, 3, 2, -1],
            [-1, 2, 0, 1]])

    res = affine_substitute_eq(f, sub_mat)

    assert np.all(res == [7, 5, 6, -5])

def test_affine_substitute_lgt():
    f = np.array([1, 2, -3])
    sub_mat_u = np.array([
            [0, 3, 2],
            [5, 3, 0]
    ])
    sub_mat_l = np.array([
            [1, 1, -2],
            [0, 2, 1]
    ])

    res_lt = affine_substitute_lt(f, sub_mat_l ,sub_mat_u)
    res_gt = affine_substitute_gt(f, sub_mat_l ,sub_mat_u)

    assert np.all(res_lt == [1, 0, 1])
    assert np.all(res_gt == [-12, -7,-4,])

def test_backsub_transform():
    in_dpoly = DeepPoly(None,
        [0, 0],
        [[0, 0], [0, 0]],
        [1, 1],
        [[0.5, 0], [0, 0.5]],
        Box([0, 2], [0, 2])
    )
    dpoly = DeepPoly(in_dpoly,
        [-0.5, 0],
        [[1, 1], [1, -1]],
        [-0.5, 0],
        [[1, 1], [1, -1]]
    )

    print(in_dpoly)
    print(dpoly)

    backsub_transform(dpoly)

    print(dpoly.l_combined())
    print(dpoly.u_combined())

    assert np.all(dpoly.l_combined() == [
        [-0.5, 0, 0],
        [-1, 0, -0.5]
    ])
    assert np.all(dpoly.u_combined() == [
        [1.5, 0.5, 0.5],
        [1, 0.5, 0]
    ])

def test_linear_transform():
    layer = nn.Linear(2, 2)
    layer.weight[:] = torch.Tensor([[-1, 2], [5, -18]])
    layer.bias[:] = torch.Tensor([7, -2])
    in_dpoly = DeepPoly(None, None, None, None, None,
        box = Box([-1, -2], [3, 1]))
    lu_exp = [[7, -1, 2], [-2, 5, -18]]

    dpoly = linear_transform(in_dpoly, layer)

    
    assert np.all(dpoly.l_combined() == lu_exp)
    assert np.all(dpoly.u_combined() == lu_exp)

def test_relu_transform():
    in_dpoly = DeepPoly(None, None, None, None, None, 
        box = Box([1, -4, -2, -3], [5, -2, 2, 5]))

    dpoly = relu_transform(in_dpoly)

    assert np.all(np.isclose(dpoly.l_combined(), [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]))
    assert np.all(np.isclose(dpoly.u_combined(), [
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 0, 0, 0.5, 0],
        [1.875, 0, 0, 0, 0.625]
    ]))

def test_normalization_transform():
    layer = Normalization('cpu')
    in_dpoly = DeepPoly(None,
        [0, 0],
        [[0, 0], [0, 0]],
        [1, 1],
        [[0.5, 0], [0, 0.5]],
        Box([0, 2], [0, 2])
    )
    exp_lu = [
        [-0.424212918, 3.245699448, 0],
        [-0.424212918, 0, 3.245699448],
    ]

    dpoly = normalization_transform(in_dpoly, layer)

    assert np.all(np.isclose(dpoly.l_combined(), exp_lu))
    assert np.all(np.isclose(dpoly.u_combined(), exp_lu))

def test_flatten_transform():
    in_dpoly = DeepPoly(None,
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        Box([[0, 1],[2, 3]], [[4, 5],[6,7]]),
        in_shape = (2, 2)
    )
    exp_luw = [
        [[1, 0], [0, 0]],
        [[0, 1], [0, 0]],
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]],
    ]
    exp_lub = [0, 0, 0, 0]

    dpoly = flatten_transform(in_dpoly)

    assert np.all(dpoly.l_bias == exp_lub)
    assert np.all(dpoly.l_weights == exp_luw)
    assert np.all(dpoly.u_bias == exp_lub)
    assert np.all(dpoly.u_weights == exp_luw)

if __name__ == "__main__":
    test_flatten_transform()