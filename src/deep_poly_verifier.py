from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from box import Box
from deep_poly import DeepPoly
from networks import FullyConnected
from deep_poly_transformations import layer_transform, backsub_transform

class DeepPolyVerifier():
    boxes: List[Box]
    dpolys: List[DeepPoly]
    true_label: int
    eps: float
    net_layers: List[nn.Module]

    def __init__(self, net: nn.Module):
        self.net_layers = self.extract_network_layers(net)

    def extract_network_layers(self, net):
        return [module
            for module in net.modules()
            if not isinstance(module, (FullyConnected, nn.Sequential))]

    def _create_input_dpoly(self, inputs, eps, input_range):
        box_l = np.maximum(input_range[0], inputs.detach().numpy() - eps)
        box_u = np.minimum(input_range[1], inputs.detach().numpy() + eps)
        # Create a deep_poly for inputs
        dpoly_shape = (*inputs.shape, 0)
        return DeepPoly(box_l,
                np.zeros(dpoly_shape),
                box_u,
                np.zeros(dpoly_shape),
                Box(box_l, box_u),
                "layer0")

    def verify(self, inputs: torch.Tensor, eps: float, 
            input_range: Tuple[float, float], true_label: int) -> bool:
        self.true_label = true_label
        self.eps = eps
        self.dpolys = [self._create_input_dpoly(inputs, eps, input_range)]

        # Main loop
        print(self.dpolys[-1])
        for i, layer in enumerate(self.net_layers, 1):
            self.dpolys.append(layer_transform(self.dpolys[-1], layer))
            if i > 1:
                # The first layer is laready expressed in terms of input
                backsub_transform(self.dpolys[i], self.dpolys[i-1])

            self.dpolys[-1].name = f"layer{len(self.dpolys) - 1}"
            print(self.dpolys[-1])

        return self.is_provable(self.dpolys[-1])
   
    def is_provable(self, dpoly: DeepPoly) -> bool:
        l = dpoly.box.l
        u = dpoly.box.u
        target_l = l[self.true_label]
        other_idx = np.arange(len(l)) != self.true_label
        max_other_score = u[other_idx].max()

        if target_l > max_other_score:
            return True
        else:
            return False


if __name__ == "__main__":
    l1 = nn.Linear(2, 2)
    l1.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l1.bias[:] = torch.Tensor([0, 0])

    l2 = nn.Linear(2, 2)
    l2.weight[:] = torch.Tensor([[1, 1], [1, -1]])
    l2.bias[:] = torch.Tensor([-0.5, 0])

    l3 = nn.Linear(2, 2)
    l3.weight[:] = torch.Tensor([[-1, 1], [0, 1]])
    l3.bias[:] = torch.Tensor([3, 0])

    # Test case 1
    net = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)
    input = torch.Tensor([0, 0])
    eps = 1
    
    verifier = DeepPolyVerifier(net, input, eps, 0)
    print(f"Verified 1: {verifier.verify()}")
    # print("Bounds:")
    # for dp in verifier.dpolys:
    #     print(dp)
    #     print("==========")
    # print()

    # Test case 2
    # net = nn.Sequential(l1, nn.ReLU(), l2)
    # input = torch.Tensor([0, 0])
    # eps = 1
    
    # verifier = DeepPolyVerifier(net, input, eps, 0)
    # print(f"Verified 1: {verifier.verify()}")