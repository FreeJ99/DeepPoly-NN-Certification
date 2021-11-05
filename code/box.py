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



def transform_box(box: Box, layer: nn.Module) -> Box:
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
        lower_bounds = b + np.sum(np.minimum(W * box.l, W * box.u), axis = 1)
        upper_bounds = b + np.sum(np.maximum(W * box.l, W * box.u), axis = 1)
        
    elif isinstance(layer, nn.ReLU):
        lower_bounds = np.maximum(box.l, 0)
        upper_bounds = np.maximum(box.u, 0)

    else:
        return None

    return Box(lower_bounds, upper_bounds)



class BoxVerifier(): 
    boxes: List[Box]
    true_label: int
    eps: float

    def __init__(self, net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int):
        self.true_label = true_label
        self.eps = eps

        net_layers = [module
            for module in net.modules()
            if not isinstance(module, (FullyConnected, nn.Sequential))]

        self.boxes = []
        # Create a box for inputs
        lower_bounds = np.maximum(inputs.detach().numpy() - eps, 0)
        upper_bounds = np.minimum(inputs.detach().numpy() + eps, 1)
        self.boxes.append(Box(lower_bounds, upper_bounds))

        for layer in net_layers:
            self.boxes.append(transform_box(self.boxes[-1], layer))


    def verify(self) -> bool:
        target_l = self.boxes[-1].l[self.true_label]
        other_idx = np.arange(len(self.boxes[-1].l)) != self.true_label
        max_other_score = self.boxes[-1].u[other_idx].max()

        if target_l > max_other_score:
            return True
        else:
            return False


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
    
    verifier = BoxVerifier(net, in1, eps1, 0)
    ver1 = verifier.verify()
    boxes1 = verifier.boxes
    print(f"Verified 1: {ver1}")
    print("Bounds:")
    for box in boxes1:
        print(box.l)
        print(box.u)
        print("==========")
    print()

    assert ver1 == True

    verifier = BoxVerifier(net, in2, eps2, 0)
    ver2 = verifier.verify()
    boxes2 = verifier.boxes
    print(f"Verified 2: {ver2}")
    print("Bounds:")
    for box in boxes2:
        print(box.l)
        print(box.u)
        print("==========")

    assert ver2 == False
