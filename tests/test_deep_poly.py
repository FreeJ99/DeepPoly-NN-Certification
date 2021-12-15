import numpy as np

from deep_poly import DeepPoly
from box import Box

def generateDpoly1(in_dpoly):
    lb = [-0.5, 0.5]
    lw = [[1, 1],
        [1, -1]]
    ub = [-0.5, 0.5]
    uw = [[1, 1],
        [1, -1]]
    return DeepPoly(in_dpoly, lb, lw, ub, uw, None)

def test_calculate_box():
    in_dpoly = DeepPoly(None, None, None, None, None, 
        Box([0, -1], [2, 3]))
    dpoly = generateDpoly1(in_dpoly)
    
    exp_l = [-1.5, -2.5]
    exp_u = [4.5, 3.5]
    
    dpoly.calculate_box()

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

if __name__ == "__main__":
    test_combined()
