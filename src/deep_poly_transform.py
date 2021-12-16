from itertools import product

import numpy as np
from torch import nn

from networks import Normalization
from deep_poly import DTYPE, DeepPoly


def anull_neg(f):
    return f * (f >= 0)


def anull_nonneg(f):
    return f * (f < 0)


def affine_expand(mat):
    exp_mat = np.vstack([np.zeros(mat.shape[1]), mat])
    exp_mat[0, 0] = 1
    return exp_mat


def affine_substitute_eq(f, sub_mat):
    """
    f (, n+1) = [a_0 a_1 ... a_n]
        z = a_0 + a_1 * y_1 + ... + a_n * y_n
    sub_mat (n, m+1) = [b | B]
        y_1 = b_1_0 + b_1_1 * y_1 + ... + b_1_m * y_m
        ...
        y_n = b_n_0 + b_n_1 * y_1 + ... + b_n_m * y_m
    """
    return f @ affine_expand(sub_mat)


def affine_substitute_lt(f, sub_mat_l, sub_mat_u):
    """
    f (, n+1) = [a_0 a_1 ... a_n]
        z <= a_0 + a_1 * y_1 + ... + a_n * y_n
    sub_mat_l (n, m+1) = [b | B]
        y_1 >= b_1_0 + b_1_1 * y_1 + ... + b_1_m * y_m
        ...
        y_n >= b_n_0 + b_n_1 * y_1 + ... + b_n_m * y_m
    sub_mat_u (n, m+1) = [b | B] 
    (b and B not the same as in sub_mat_l)
        y_1 <= b_1_0 + b_1_1 * y_1 + ... + b_1_m * y_m
        ...
        y_n <= b_n_0 + b_n_1 * y_1 + ... + b_n_m * y_m
    """
    return (anull_neg(f) @ affine_expand(sub_mat_u) +
        anull_nonneg(f) @ affine_expand(sub_mat_l))

def affine_substitute_gt(f, sub_mat_l, sub_mat_u):
    """
    f (, n+1) = [a_0 a_1 ... a_n]
        z >= a_0 + a_1 * y_1 + ... + a_n * y_n
    sub_mat_l (n, m+1) = [b | B]
        y_1 >= b_1_0 + b_1_1 * y_1 + ... + b_1_m * y_m
        ...
        y_n >= b_n_0 + b_n_1 * y_1 + ... + b_n_m * y_m
    sub_mat_u (n, m+1) = [b | B]
    (b and B not the same as in sub_mat_l)
        y_1 <= b_1_0 + b_1_1 * y_1 + ... + b_1_m * y_m
        ...
        y_n <= b_n_0 + b_n_1 * y_1 + ... + b_n_m * y_m
    """
    return (anull_neg(f) @ affine_expand(sub_mat_l) +
        anull_nonneg(f) @ affine_expand(sub_mat_u))


def backsub_transform(dpoly: DeepPoly):
    '''Expresses cur_dpoly in terms of the inputs to in_dpoly.
    
    Works in place.
    '''
    for i in range(dpoly.n_neur()):
        neur = dpoly.get_neur(i)
        in_dpoly = dpoly.in_dpoly
        new_l = affine_substitute_gt(neur.l, in_dpoly.l_combined(), in_dpoly.u_combined())
        new_u = affine_substitute_lt(neur.u, in_dpoly.l_combined(), in_dpoly.u_combined())
        dpoly.update_neur(i, new_l[0], new_l[1:], new_u[0], new_u[1:])

    dpoly.in_dpoly = dpoly.in_dpoly.in_dpoly
    dpoly.calculate_box()


def layer_transform(in_dpoly: DeepPoly, layer: nn.Module):
    if isinstance(layer, Normalization):
        return normalization_transform(in_dpoly, layer)
    elif isinstance(layer, nn.Flatten):
        return flatten_transform(in_dpoly)
    elif isinstance(layer, nn.Linear):
       return linear_transform(in_dpoly, layer)   
    elif isinstance(layer, nn.ReLU):
        return relu_transform(in_dpoly)
    else:
        raise NotImplementedError

def normalization_transform(dpoly: DeepPoly, layer: Normalization):
    m = layer.mean.detach().numpy()
    s = layer.sigma.detach().numpy()

    l_bias = np.full((dpoly.n_neur(), ), - m / s)
    # l_weights[i,j,k,i,j,k] = 1, number of dimensions might vary
    l_weights = np.zeros((dpoly.n_neur(), dpoly.n_neur()))
    l_weights_idx = 2 * tuple(
                map(list,
                zip(
                *product(
                *map(range, (dpoly.n_neur(),))
                ))))
    l_weights[l_weights_idx] = 1 / s

    u_bias = l_bias.copy()
    u_weights = l_weights.copy()

    return DeepPoly(dpoly, l_bias, l_weights, u_bias, u_weights)


def linear_transform(in_dpoly: DeepPoly, layer: nn.Linear):
    W = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    l_bias = b.copy()
    l_weights = W.copy()
    u_bias = b.copy()
    u_weights = W.copy()

    return DeepPoly(in_dpoly, l_bias, l_weights, u_bias, u_weights)


def relu_transform(in_dpoly: DeepPoly):
    n_neur = in_dpoly.n_neur()
    l_bias = np.zeros(n_neur, dtype = DTYPE)
    l_weights = np.zeros((n_neur, n_neur), dtype = DTYPE)
    u_bias = l_bias.copy()
    u_weights = l_weights.copy()

    neg_idx = in_dpoly.box.u <= 0
    # all values already set to 0

    pos_idx = in_dpoly.box.l >= 0
    # l_bias[pos_idx] = in_dpoly.l_bias[pos_idx]
    # l_bias already set to 0
    l_weights[pos_idx] = np.eye(n_neur)[pos_idx]
    # u_bias[pos_idx] = in_dpoly.u_bias[pos_idx]
    # u_bias already set to 0
    u_weights[pos_idx] = np.eye(n_neur)[pos_idx]

    crossing_idx = ~(neg_idx | pos_idx)
    slope = (in_dpoly.box.u) / (in_dpoly.box.u - in_dpoly.box.l)
    y_intercept = - slope * in_dpoly.box.l
    # l_bias already set to 0
    # l_weights already set to 0
    u_bias[crossing_idx] = y_intercept[crossing_idx]
    u_weights[crossing_idx] = np.diag(slope)[crossing_idx]

    return DeepPoly(in_dpoly, l_bias, l_weights, u_bias, u_weights)


def flatten_transform(in_dpoly: DeepPoly):
    size = in_dpoly.layer_size()
    shape = in_dpoly.layer_shape
    
    l_bias = np.zeros(size)
    # k-th neuron in the flatten layer corresponds to
    # the k-th neuron in the input layer
    l_weights = np.zeros((size, *shape))
    l_weights_view = l_weights.reshape(size, -1)
    np.fill_diagonal(l_weights_view, 1)

    u_bias = np.zeros(size)
    u_weights = l_weights.copy()

    return DeepPoly(in_dpoly, l_bias, l_weights, u_bias, u_weights,
        in_shape = shape)