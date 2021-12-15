from deep_poly_transform import *

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

if __name__ == "__main__":
    test_backsub_transform()