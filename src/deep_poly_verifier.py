from copy import deepcopy
from typing import List, Tuple

import numpy as np
from numpy.lib.function_base import diff
import torch
from torch import nn

from box import Box
from deep_poly import DeepPoly
from networks import FullyConnected
from deep_poly_transform import layer_transform, backsub_transform


def extract_network_layers(net):
    return [module
            for module in net.modules()
            if not isinstance(module, (FullyConnected, nn.Sequential))]


def _create_input_dpoly(inputs, eps, input_range):
    box_l = np.maximum(input_range[0], inputs - eps)
    box_u = np.minimum(input_range[1], inputs + eps)
    # Create a deep_poly for inputs
    dpoly_shape = (*inputs.shape, 0)
    return DeepPoly(
        None,
        box_l,
        np.zeros(dpoly_shape),
        box_u,
        np.zeros(dpoly_shape),
        Box(box_l, box_u)
    )


def backsubstitute(start_dpoly):
    if start_dpoly.in_dpoly is None:
        return

    curr_dp = start_dpoly
    while curr_dp.in_dpoly.in_dpoly is not None:
        curr_dp = backsub_transform(curr_dp)
    start_dpoly.box = curr_dp.box


def is_provable(dpoly: DeepPoly, true_label, verbose=False) -> bool:
    # Sets up a layer that gives the desired differences
    n_neur = dpoly.layer_size()
    tmp = np.zeros((n_neur, n_neur))
    tmp[:, true_label] = 1
    weight = tmp - np.eye(n_neur)
    bias = np.zeros((n_neur))

    layer = nn.Linear(n_neur, n_neur)
    layer.weight[:] = torch.Tensor(weight)
    layer.bias[:] = torch.Tensor(bias)

    # Creates a dpoly of differences
    diff_dpoly = layer_transform(dpoly, layer)
    diff_dpoly.name = "diff"
    backsubstitute(diff_dpoly)
    # x_true_label - x_true_label = 0, which we don't want
    diff_dpoly.box.l[true_label] = np.inf

    if verbose:
        print(diff_dpoly)
    return np.all(diff_dpoly.box.l > 0)


def preprocess_inputs(inputs):
    inputs = inputs.detach().numpy()
    if inputs.ndim > 2:
        img_width = int(inputs.size ** 0.5)
        inputs = inputs.reshape((img_width, img_width))

    return inputs


class DeepPolyVerifier():
    """
    Batches currently not supported.
    """
    dpolys: List[DeepPoly]
    net_layers: List[nn.Module]

    def __init__(self, net: nn.Module):
        self.net_layers = extract_network_layers(net)

    def verify(self, inputs: torch.Tensor, eps: float,
               true_label: int,
               input_range=[-np.inf, +np.inf],
               verbose=False) -> bool:

        inputs = preprocess_inputs(inputs)
        self.dpolys = [_create_input_dpoly(inputs, eps, input_range)]
        if verbose:
            self.dpolys[-1].name = "input"

        # Main loop
        if verbose:
            print(self.dpolys[-1])

        for _, layer in enumerate(self.net_layers, 1):
            self.dpolys.append(layer_transform(self.dpolys[-1], layer))
            # The first layer is laready expressed in terms of the input
            if verbose:
                self.dpolys[-1].name = f"layer{len(self.dpolys) - 1}_{type(layer)}"
                print(self.dpolys[-1])

            if isinstance(layer, nn.Linear):
                backsubstitute(self.dpolys[-1])

                if verbose:
                    print("After backsub")
                    print(self.dpolys[-1].box, end="\n\n\n")

        return is_provable(self.dpolys[-1], true_label, verbose)
