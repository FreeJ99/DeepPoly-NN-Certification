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
    box_l = np.maximum(input_range[0], inputs.detach().numpy() - eps)
    box_u = np.minimum(input_range[1], inputs.detach().numpy() + eps)
    # Create a deep_poly for inputs
    dpoly_shape = (*inputs.shape, 0)
    return DeepPoly(
            None,
            box_l,
            np.zeros(dpoly_shape),
            box_u,
            np.zeros(dpoly_shape),
            Box(box_l, box_u),
            "layer0")

def is_provable(dpoly: DeepPoly, true_label, verbose = False) -> bool:
    n_neur = dpoly.n_neur()
    tmp = np.zeros((n_neur, n_neur))
    tmp[:, true_label] = 1
    weight = tmp - np.eye(n_neur)
    bias = np.zeros((n_neur))

    layer = nn.Linear(n_neur, n_neur)
    layer.weight[:] = torch.Tensor(weight)
    layer.bias[:] = torch.Tensor(bias)
    
    diff_dpoly = layer_transform(dpoly, layer)
    diff_dpoly.name = "diff"
    backsub_transform(diff_dpoly)
    # x_true_label - x_true_label = 0, which we don't want
    diff_dpoly.box.l[true_label] = np.inf

    if verbose:
        print(diff_dpoly)
    return np.all(diff_dpoly.box.l > 0)


class DeepPolyVerifier():
    """
    In this implementation every layer gets immideately 
    represented in terms of the input.
    """
    dpolys: List[DeepPoly]
    net_layers: List[nn.Module]

    def __init__(self, net: nn.Module):
        self.net_layers = extract_network_layers(net)

    def verify(self, inputs: torch.Tensor, eps: float,
            true_label: int, 
            input_range = [-np.inf, +np.inf],
            verbose = False) -> bool:
        self.dpolys = [_create_input_dpoly(inputs, eps, input_range)]

        # Main loop
        if verbose:
            print(self.dpolys[-1])
        for i, layer in enumerate(self.net_layers, 1):
            self.dpolys.append(layer_transform(self.dpolys[-1], layer))
            # The first layer is laready expressed in terms of the input
            if verbose:
                self.dpolys[-1].name = f"layer{len(self.dpolys) - 1}"
                print(self.dpolys[-1])

            if isinstance(layer, nn.Linear):
                self.backsubstitute(i)

                if verbose:
                    print("After backsub")
                    print(self.dpolys[-1])

        return is_provable(self.dpolys[-1], true_label, verbose)

    def backsubstitute(self, i):
        dp = self.dpolys[-1]
        cp_dpoly = DeepPoly(dp.in_dpoly,
            dp.l_bias.copy(),
            dp.l_weights.copy(),
            dp.u_bias.copy(),
            dp.u_weights.copy()
        )
        for _ in reversed(range(1, i)):
            backsub_transform(cp_dpoly)
        self.dpolys[-1].box = cp_dpoly.box