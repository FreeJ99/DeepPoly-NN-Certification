import numpy as np

from deep_poly import DeepPoly
from box import Box

def test_calculate_box():
    in_dpoly = DeepPoly(None, None, None, None, None, 
        Box(np.array([0, -1]), np.array([2, 3])))
    lb = np.array([-0.5, 0.5])
    lw = np.array([[1, 1],
                   [1, -1]])
    ub = np.array([-0.5, 0.5])
    uw = np.array([[1, 1],
                   [1, -1]])
    dpoly = DeepPoly(in_dpoly, lb, lw, ub, uw, None)
    
    exp_l = [-1.5, -2.5]
    exp_u = [4.5, 3.5]
    
    dpoly.calculate_box()

    assert np.all(np.isclose(dpoly.box.l, exp_l))
    assert np.all(np.isclose(dpoly.box.u, exp_u))

if __name__ == "__main__":
    test_calculate_box()