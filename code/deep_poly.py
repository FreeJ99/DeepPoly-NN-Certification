from typing import List, Tuple
from itertools import product

import numpy as np
from numpy.core.defchararray import upper
from numpy.core.numeric import cross
import torch
from torch import nn
from torch.nn.modules.activation import ReLU

from networks import Normalization, FullyConnected
from box import Box, transform_box
from verifier import INPUT_SIZE

class DeepPoly():
    l_bias: np.ndarray
    l_weights: np.ndarray
    u_bias: np.ndarray
    u_weights: np.ndarray
    box: Box
    input_shape: Tuple[int]
    layer_shape: Tuple[int]

    def __init__(self, l_bias, l_weights, u_bias, u_weights, box):
        self.l_bias = l_bias
        self.l_weights = l_weights
        self.u_bias = u_bias
        self.u_weights = u_weights
        self.box = box
        self.layer_shape = box.l.shape
        self.input_shape = l_weights.shape[box.l.ndim:]
        print(self.input_shape)
    
    def __repr__(self):
        return "DPoly:\n{0}\n{1}\n{2}\n{3}\nBox:\n{4}\n{5}\n\n"\
            .format(self.l_bias, self.l_weights, self.u_bias, self.u_weights, 
                self.box.l, self.box.u)


def transform_deep_poly(in_dpoly: DeepPoly, layer: nn.Module):
    box = transform_box(in_dpoly.box, layer)
    
    if isinstance(layer, Normalization):
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

    elif isinstance(layer, nn.Flatten):
        pass

    elif isinstance(layer, nn.Linear):
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        l_bias = b.copy()
        l_weights = W.copy()
        u_bias = b.copy()
        u_weights = W.copy()
        
    elif isinstance(layer, nn.ReLU):
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


class DeepPolyVerifier():
    boxes: List[Box]
    true_label: int
    eps: float
    net_layers: List[nn.Module] # TODO eliminate

    def __init__(self, net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int):
        self.true_label = true_label
        self.eps = eps

        self.net_layers = [module
            for module in net.modules()
            if not isinstance(module, (FullyConnected, nn.Sequential))]

        self.dpolys = []
        # Create a box for inputs
        # box_l = np.maximum(inputs.detach().numpy() - eps, 0)
        # box_u = np.minimum(inputs.detach().numpy() + eps, 1)
        box_l = inputs.detach().numpy() - eps
        box_u = inputs.detach().numpy() + eps
        # Create a deep_poly for inputs
        dpoly_shape = (*inputs.shape, 0)
        self.dpolys.append(DeepPoly(box_l,
                                    np.zeros(dpoly_shape),
                                    box_u,
                                    np.zeros(dpoly_shape),
                                    Box(box_l, box_u)))

        # print(self.dpolys[-1])
        for layer in self.net_layers:
            self.dpolys.append(transform_deep_poly(self.dpolys[-1], layer))
            # print(self.dpolys[-1])

    def verify(self) -> bool:
        # deep poly abstract bounds of the final layernp.array()
        curr_dpoly = self.dpolys[-1]
        print(curr_dpoly)
        for i in reversed(range(1, len(self.net_layers) - 1)):
            if isinstance(self.net_layers[i], nn.Flatten):
                break
            # calculate the new representation
            curr_dpoly = self.backsubstitute(curr_dpoly,
                                            self.dpolys[i])
            print(curr_dpoly)

            # Recalculate box constraints
            lW = curr_dpoly.l_weights
            lb = curr_dpoly.l_bias
            uW = curr_dpoly.u_weights
            ub = curr_dpoly.u_bias
            prev_box = self.dpolys[i-1].box
            box_l = lb + np.sum(np.minimum(lW * prev_box.l, lW * prev_box.u), axis = 1)
            box_u = ub + np.sum(np.maximum(uW * prev_box.l, uW * prev_box.u), axis = 1)
            curr_dpoly.box = Box(box_l, box_u)

            if self.is_provable(curr_dpoly):
                break

        return self.is_provable(curr_dpoly)

    def backsubstitute(self, dpoly: DeepPoly, prev_dpoly: DeepPoly):
        l_bias = np.empty(dpoly.layer_shape)
        l_weights = np.empty((*dpoly.layer_shape, *prev_dpoly.input_shape))
        u_bias = l_bias.copy()
        u_weights = l_weights.copy()

        for neur in range(dpoly.layer_shape[0]): # TODO 1D dependant
            # Select lower or upper bounds from the previous layer
            # depending on the weights in the current neuron
            
            # Lower dpoly bound
            prev_l_tmp_bias = (prev_dpoly.l_bias * (dpoly.l_weights[neur] >= 0)) +\
                            (prev_dpoly.u_bias * (dpoly.l_weights[neur] < 0))
            prev_l_tmp_weights = (prev_dpoly.l_weights * (dpoly.l_weights[neur] >= 0)) +\
                            (prev_dpoly.u_weights * (dpoly.l_weights[neur] < 0))
            l_bias[neur] = dpoly.l_bias[neur] + dpoly.l_weights[neur] @ prev_l_tmp_bias.T
            l_weights[neur] = dpoly.l_weights[neur] @ prev_l_tmp_weights

            # Upper dpoly bound
            prev_u_tmp_bias = (prev_dpoly.u_bias * (dpoly.u_weights[neur] >= 0)) +\
                            (prev_dpoly.l_bias * (dpoly.u_weights[neur] < 0))
            prev_u_tmp_weights = (prev_dpoly.u_weights * (dpoly.u_weights[neur] >= 0)) +\
                            (prev_dpoly.l_weights * (dpoly.u_weights[neur] < 0))
            u_bias[neur] = dpoly.u_bias[neur] + dpoly.u_weights[neur] @ prev_u_tmp_bias.T
            u_weights[neur] = dpoly.u_weights[neur] @ prev_u_tmp_weights

        return DeepPoly(l_bias, l_weights, u_bias, u_weights, dpoly.box) # TODO keeping old box for now

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

    net = nn.Sequential(l1, nn.ReLU(), l2, nn.ReLU(), l3)

    # Test case 1
    input = torch.Tensor([0, 0])
    eps = 1
    
    verifier = DeepPolyVerifier(net, input, eps, 0)
    print(f"Verified 1: {verifier.verify()}")
    # print("Bounds:")
    # for dp in verifier.dpolys:
    #     print(dp)
    #     print("==========")
    # print()