from typing import List, Tuple

import numpy as np
from numpy.core.defchararray import upper
import torch
from torch import nn
from torch.nn.modules.activation import ReLU

from networks import Normalization, FullyConnected


class Box():
    '''Represents the constraints for a single layer in a network.'''

    l: np.ndarray
    u: np.ndarray

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.l = np.array(lower_bounds)
        self.u = np.array(upper_bounds)

    def flatten(self):
        return self.l.flatten(), self.u.flatten()

    def __repr__(self):
        return (f"Box l = {self.l}\n" +
                f"Box u = {self.u}")


def transform_box(box: Box, layer: nn.Module) -> Box:
    '''Transforms      '''
    if isinstance(layer, Normalization):
        m = layer.mean.detach().numpy()
        s = layer.sigma.detach().numpy()
        lower_bounds = (box.l - m) / s
        upper_bounds = (box.u - m) / s

    elif isinstance(layer, nn.Flatten):
        lower_bounds = box.l.flatten()
        upper_bounds = box.u.flatten()

    elif isinstance(layer, nn.Linear):
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        lower_bounds = b + np.sum(np.minimum(W * box.l, W * box.u), axis=1)
        upper_bounds = b + np.sum(np.maximum(W * box.l, W * box.u), axis=1)

    elif isinstance(layer, nn.ReLU):
        lower_bounds = np.maximum(box.l, 0)
        upper_bounds = np.maximum(box.u, 0)

    else:
        raise NotImplementedError

    return Box(lower_bounds, upper_bounds)


class BoxVerifier():
    '''Runs the Verification prucedure using Box and stores the results.'''
    boxes: List[Box]

    def __init__(self, net: nn.Module):
        self.net_layers = [module
                           for module in net.modules()
                           if not isinstance(module, (FullyConnected, nn.Sequential))]

    def verify(self, inputs: torch.Tensor, eps: float, true_label: int,
               input_range=[-np.inf, +np.inf]) -> bool:
        self.boxes = []
        # Create a box for inputs
        box_l = np.maximum(input_range[0], inputs - eps)
        box_u = np.minimum(input_range[1], inputs + eps)
        self.boxes.append(Box(box_l, box_u))

        for layer in self.net_layers:
            self.boxes.append(transform_box(self.boxes[-1], layer))

        target_l = self.boxes[-1].l[true_label]
        other_idx = np.arange(len(self.boxes[-1].l)) != true_label
        max_other_score = self.boxes[-1].u[other_idx].max()

        if target_l > max_other_score:
            return True
        else:
            return False
