from itertools import product

import numpy as np
from torch import nn

from networks import Normalization
from box import transform_box
from deep_poly import DeepPoly
from box import Box

def get_positive_mask():
    raise NotImplementedError

def get_negative_mask():
    raise NotImplementedError

def substitute_neuron_u_bounds(coef,  in_l_bounds, in_u_bounds):
    """Performs variable substitution on an inequality.
    
    Args:
        cur_matrix: 
    """
    raise NotImplementedError
    prev_u_tmp_bias = (in_dpoly.u_bias * (cur_dpoly.u_weights[neur] >= 0)) +\
                    (in_dpoly.l_bias * (cur_dpoly.u_weights[neur] < 0))
    prev_u_tmp_weights = (in_dpoly.u_weights * (cur_dpoly.u_weights[neur] >= 0)) +\
                    (in_dpoly.l_weights * (cur_dpoly.u_weights[neur] < 0))
    u_bias[neur] = cur_dpoly.u_bias[neur] + cur_dpoly.u_weights[neur] @ prev_u_tmp_bias.T
    u_weights[neur] = cur_dpoly.u_weights[neur] @ prev_u_tmp_weights

def substitute_neuron_l_bounds(coef,  in_l_bounds, in_u_bounds):
    raise NotImplementedError
    # Lower dpoly bound
    prev_l_tmp_bias = (in_dpoly.l_bias * (cur_dpoly.l_weights[neur] >= 0)) +\
                    (in_dpoly.u_bias * (cur_dpoly.l_weights[neur] < 0))
    prev_l_tmp_weights = (in_dpoly.l_weights * (cur_dpoly.l_weights[neur] >= 0)) +\
                    (in_dpoly.u_weights * (cur_dpoly.l_weights[neur] < 0))
    l_bias[neur] = cur_dpoly.l_bias[neur] + cur_dpoly.l_weights[neur] @ prev_l_tmp_bias.T
    l_weights[neur] = cur_dpoly.l_weights[neur] @ prev_l_tmp_weights


def substitute_layer_u_bounds(u_combined, in_l_combined, in_u_combined):
    raise NotImplementedError
    new_rows = []
    for row in range(cur_dpoly.layer_shape[0]): # TODO 1D dependant
        # Select lower or upper bounds from the previous layer
        # depending on the weights in the current neuron
        in_u_mask = get_positive_mask()
        in_l_mask = get_negetive_mask()
        combined_in = (in_dpoly.u_combined() * in_u_mask
                    + in_dpoly.l_combined() * in_l_mask)
        
        # Substitutes the new input variables
        new_row = row @ combined_in
        new_rows.append(new_row)

    new_rows = np.vstack(new_rows)
    in_dpoly.u_weights = new_rows[:, 1:]
    in_dpoly.u_bias = new_rows[:, 0]

def substitute_layer_l_bounds(coef_matrix, in_l_coef, in_u_coef):
    raise NotImplementedError

def backsub_transform(cur_dpoly: DeepPoly):
    '''Expresses cur_dpoly in terms of the inputs to in_dpoly.
    
    Works in place.
    '''
    cur_dpoly.u_weights, cur_dpoly.u_bias = substitute_layer_u_bounds(
        cur_dpoly.u_combined(),
        cur_dpoly.in_dpoly.l_combined_with_ones(),
        cur_dpoly.in_dpoly.u_combined_with_ones()
    )
    cur_dpoly.l_weights, cur_dpoly.l_bias = substitute_layer_l_bounds()

    cur_dpoly.calculate_box()


def layer_transform(in_dpoly: DeepPoly, layer: nn.Module):
    if isinstance(layer, Normalization):
        return normalization_transform(in_dpoly, layer)
    elif isinstance(layer, nn.Flatten):
        raise NotImplementedError
    elif isinstance(layer, nn.Linear):
       return linear_transform(in_dpoly, layer)   
    elif isinstance(layer, nn.ReLU):
        return relu_transform(in_dpoly, layer)
    else:
        raise NotImplementedError

def normalization_transform(in_dpoly: DeepPoly, layer: Normalization):
    raise NotImplementedError

    box = transform_box(in_dpoly.box, layer)
    
    m = layer.mean.detach().numpy()
    s = layer.sigma.detach().numpy()

    l_bias = np.full(in_dpoly.layer_shape, - m / s)
    # l_weights[i,j,k,i,j,k] = 1, number of dimensions might vary
    l_weights = np.zeros(2 * in_dpoly.layer_shape)
    l_weights_idx = 2 * list(
                map(lambda x: list(x),
                zip(
                *product(
                *map(range, in_dpoly.layer_shape)))))
    l_weights[l_weights_idx] = 1 / s

    u_bias = l_bias.copy()
    u_weights = l_weights.copy()

    return DeepPoly(l_bias, l_weights, u_bias, u_weights, box)


def linear_transform(in_dpoly: DeepPoly, layer: nn.Linear):
    raise NotImplementedError

    box = transform_box(in_dpoly.box, layer)

    W = layer.weight.detach().numpy()
    b = layer.bias.detach().numpy()
    l_bias = b.copy()
    l_weights = W.copy()
    u_bias = b.copy()
    u_weights = W.copy()

    return DeepPoly(l_bias, l_weights, u_bias, u_weights, box)


def relu_transform(in_dpoly: DeepPoly, layer: nn.Linear):
    raise NotImplementedError

    box = transform_box(in_dpoly.box, layer)

    l_bias = np.zeros(in_dpoly.layer_shape)
    l_weights = np.zeros(2 * in_dpoly.layer_shape)
    u_bias = l_bias.copy()
    u_weights = l_weights.copy()

    neg_idx = in_dpoly.box.u <= 0
    # all values already set to 0

    pos_idx = in_dpoly.box.l >= 0
    l_bias[pos_idx] = in_dpoly.l_bias[pos_idx]
    l_weights[pos_idx] = np.eye(in_dpoly.layer_shape[0])[pos_idx]
    u_bias[pos_idx] = in_dpoly.u_bias[pos_idx]
    u_weights[pos_idx] = l_weights[pos_idx]

    crossing_idx = ~(neg_idx | pos_idx)
    slope = (in_dpoly.box.u) / (in_dpoly.box.u - in_dpoly.box.l)
    y_intercept = - slope * in_dpoly.box.l
    # l_bias already set to 0
    # l_weights already set to 0
    u_bias[crossing_idx] = y_intercept[crossing_idx]
    u_weights[crossing_idx] = np.diag(slope)[crossing_idx]

    return DeepPoly(l_bias, l_weights, u_bias, u_weights, box)