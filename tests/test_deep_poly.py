import numpy as np

from deep_poly import DeepPoly
from box import Box


def test_calculate_box():
    in_dpoly = DeepPoly(None, None, None, None, None,
        Box([0, -1], [2, 3]))
    lb = [-0.5, 0.5]
    lw = [[1, 1],
        [1, -1]]
    ub = [-0.5, 0.5]
    uw = [[1, 1],
        [1, -1]]
    dpoly = DeepPoly(in_dpoly, lb, lw, ub, uw)

    exp_l = [-1.5, -2.5]
    exp_u = [4.5, 3.5]

    assert np.all(np.isclose(dpoly.box.l, exp_l))
    assert np.all(np.isclose(dpoly.box.u, exp_u))

def test_combined_and_ones():
    lb = [1, 2, 2]
    lw = [[3, 4],
        [5, 6],
        [7, 8]]
    ub = [9, 10, 10]
    uw = [[11, 12],
        [13, 14],
        [15, 16]]
    dpoly = DeepPoly(None, lb, lw, ub, uw, None)

    exp_l_comb = [[1, 3, 4],
                [2, 5, 6],
                [2, 7, 8]]
    exp_u_comb1 = [[1, 1, 1],
                [9, 11, 12],
                [10, 13, 14],
                [10, 15, 16]]

    l_comb = dpoly.l_combined()
    u_comb1 = dpoly.u_combined_ones()

    assert np.all(l_comb == exp_l_comb)
    assert np.all(u_comb1 == exp_u_comb1)

def test_calculate_box_2d():
    in_dpoly = DeepPoly(None,
        None, None, None, None,
        Box([[-1, -1], [-1, -1]], [[1, 1], [1, 1]]),
        layer_shape = (2, 2)
    )

    # 2x2x2x2
    l_weights = [
        [
            [[-1, 0], [1, 0]],
            [[0, 1], [1, 0]]
        ],
        [
            [[-1, 1], [1, 0]],
            [[0, 0], [0, -1]]
        ]
    ]

    dpoly = DeepPoly(in_dpoly,
        [[-1, 1], [2, 0]],
        l_weights,
        [[-1, 1], [2, 0]],
        l_weights.copy(),
        layer_shape = (2, 2)
    )
    exp_box_l = [
        [-3, -1],
        [-1, -1]
    ]
    exp_box_u = [
        [1, 3],
        [5, 1]
    ]

    assert np.all(np.isclose(dpoly.box.l, exp_box_l))
    assert np.all(np.isclose(dpoly.box.u, exp_box_u))

if __name__ == "__main__":
    test_calculate_box_2d()
