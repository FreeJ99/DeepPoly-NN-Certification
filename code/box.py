from typing import List, Tuple

import numpy as np
from numpy.core.defchararray import upper
import torch
from torch import nn
from torch.nn.modules.activation import ReLU

from networks import Normalization, FullyConnected

class Box():
    l: np.ndarray
    u: np.ndarray

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray):
        self.l = lower_bounds
        self.u = upper_bounds


def transform_box_layer(box: Box, layer: nn.Module) -> Box:
    # Box abstract transformers
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
        lower_bounds = np.sum(np.minimum(W * box.l, W * box.u), axis = 1)
        upper_bounds = np.sum(np.maximum(W * box.l, W * box.u), axis = 1)

        if layer.bias != None:
            b = layer.bias.detach().numpy()
            lower_bounds += b
            upper_bounds += b
        
    elif isinstance(layer, nn.ReLU):
        lower_bounds = np.maximum(box.l, 0)
        upper_bounds = np.maximum(box.u, 0)

    else:
        return None

    return Box(lower_bounds, upper_bounds)

def box_verify_net(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> Tuple[bool, List[Box]]:
    net_layers = [module
        for module in net.modules()
        if not isinstance(module, (FullyConnected, nn.Sequential))]

    boxes = []
    # Create a box for inputs
    lower_bounds = np.maximum(inputs.detach().numpy() - eps, 0)
    upper_bounds = np.minimum(inputs.detach().numpy() + eps, 1)
    boxes.append(Box(lower_bounds, upper_bounds))

    for layer in net_layers:
        boxes.append(transform_box_layer(boxes[-1], layer))

    target_l = boxes[-1].l[true_label]
    other_idx = np.arange(len(boxes[-1].l)) != true_label
    max_other_score = boxes[-1].u[other_idx].max()

    if target_l > max_other_score:
        return (True, boxes)
    else:
        return (False, boxes)

if __name__ == "__main__":
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])
    relu1 = nn.ReLU()
    l2 = nn.Linear(2, 2)
    l2.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l2.bias[:] = torch.Tensor([0.5, -0.5])
    net = nn.Sequential(l1, relu1, l2)

    in1 = torch.Tensor([0.15, 0.25])
    eps1 = 0.15
    in2 = torch.Tensor([0.3, 0.4])
    eps2 = 0.3
    
    ver1, boxes1 = box_verify_net(net, in1, eps1, 0)
    print(f"Verified 1: {ver1}")
    print("Bounds:")
    for box in boxes1:
        print(box.l)
        print(box.u)
        print("==========")
    print()

    assert ver1 == True

    ver2, boxes2 = box_verify_net(net, in2, eps2, 0)
    print(f"Verified 2: {ver2}")
    print("Bounds:")
    for box in boxes2:
        print(box.l)
        print(box.u)
        print("==========")

    assert ver2 == False

