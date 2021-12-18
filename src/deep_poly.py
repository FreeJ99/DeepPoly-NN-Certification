from typing import Tuple
from collections import namedtuple
from itertools import product

import numpy as np
from box import Box

DTYPE = np.float64

class DeepPoly():
    l_bias: np.ndarray
    l_weights: np.ndarray
    u_bias: np.ndarray
    u_weights: np.ndarray
    box: Box
    # in_dpoly: DeepPoly # TODO 
    name: str # useful for debugging
    layer_shape: Tuple[int]
    in_shape: Tuple[int]

    def __init__(self, in_dpoly, l_bias, l_weights, u_bias, u_weights, box = None,
            layer_shape = None, name = ""):
        self.l_bias = np.array(l_bias, dtype = DTYPE)
        self.l_weights = np.array(l_weights, dtype = DTYPE)
        self.u_bias = np.array(u_bias, dtype = DTYPE)
        self.u_weights = np.array(u_weights, dtype = DTYPE)
        self.in_dpoly = in_dpoly
        self.name = name

        # infers layer_shape
        if layer_shape is not None:
            self.layer_shape = layer_shape
        elif l_bias is not None:
            self.layer_shape = self.l_bias.shape
        elif box is not None:
            self.layer_shape = box.l.shape
        else:
            raise Exception("Layer's shape can't be infered.")

        # infers in_shape
        if l_weights is not None:
            layer_ndim = len(self.layer_shape)
            self.in_shape = self.l_weights.shape[layer_ndim:]
        elif in_dpoly is not None:
            self.in_shape = in_dpoly.layer_shape
        else:
            print("Warning: Input's shape can't be infered.")

        if box is None:
            self.calculate_box()
        else:
            self.box = box

    def calculate_box(self):
        if self.in_dpoly is None:
            self.box = None
            return

        lb, lW, ub, uW = self.biflatten()
        prev_box_l, prev_box_u = self.in_dpoly.box.flatten()
        box_l = lb + np.sum(np.minimum(lW * prev_box_l, lW * prev_box_u), 
            axis = 1)
        box_u = ub + np.sum(np.maximum(uW * prev_box_l, uW * prev_box_u), 
            axis = 1)
        self.box = Box(box_l.reshape(self.layer_shape), box_u.reshape(self.layer_shape))

    def biflatten(self):
        lb = self.l_bias.flatten()
        lW = self.l_weights.reshape(self.layer_size(), self.in_size())
        ub = self.u_bias.flatten()
        uW = self.u_weights.reshape(self.layer_size(), self.in_size())
        return lb, lW, ub, uW

    def l_combined(self) -> np.ndarray:
        """Merges weights and bias."""
        lb, lW, _, _ = self.biflatten()
        return np.hstack([np.expand_dims(lb, 1), lW])

    def u_combined(self) -> np.ndarray:
        _, _, ub, uW = self.biflatten()
        return np.hstack([np.expand_dims(ub, 1), uW])

    def l_combined_ones(self) -> np.ndarray:
        l_combined = self.l_combined()
        return np.vstack([np.ones(l_combined.shape[1]), l_combined])

    def u_combined_ones(self) -> np.ndarray:
        u_combined = self.u_combined()
        return np.vstack([np.ones(u_combined.shape[1]), u_combined])

    def get_neur(self, idx):
        AbstractNeuron = namedtuple('AbstractNeuron', 'l u')
        return AbstractNeuron(
            l = self.l_combined()[idx],
            u = self.u_combined()[idx]
        )

    def layer_size(self):
        return int(np.product(self.layer_shape))
        
    def in_size(self):
        return np.product(self.in_shape)

    def update_neur(self, idx, new_lb, new_lw, new_ub, new_uw):
        self.l_bias[idx] = new_lb
        self.l_weights[idx] = new_lw
        self.u_bias[idx] = new_ub
        self.u_weights[idx] = new_uw

    def layer_iterator(self):
        return product(*map(range, self.layer_shape))

    def __repr__(self):
        lines = [f"DPoly {self.name} | shape {self.layer_shape}"]
        for idx in self.layer_iterator():
            lines.append(f"neur{list(idx)}:")
            lines.append(f">=\n{self.l_weights[idx]} + {self.l_bias[idx]}")
            lines.append(f"<=\n{self.u_weights[idx]} + {self.u_bias[idx]}")
            lines.append(f"l = {self.box.l[idx]}")
            lines.append(f"u = {self.box.u[idx]}")
            lines.append("\n")

        return "\n".join(lines)
    