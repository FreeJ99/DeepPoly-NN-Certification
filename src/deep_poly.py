from typing import Tuple
from collections import namedtuple

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

    def __init__(self, in_dpoly, l_bias, l_weights, u_bias, u_weights, box = None,
            layer_shape = None, name = ""):
        self.l_bias = np.array(l_bias, dtype = DTYPE)
        self.l_weights = np.array(l_weights, dtype = DTYPE)
        self.u_bias = np.array(u_bias, dtype = DTYPE)
        self.u_weights = np.array(u_weights, dtype = DTYPE)
        self.in_dpoly = in_dpoly
        self.name = name

        # layer shape has to be infered
        # input_shape is optional, but a box has to be given if input_shape isn't
        if layer_shape is not None and l_weights is not None:
            self.layer_shape = layer_shape
            layer_ndim = len(layer_shape)
            self.in_shape = self.l_weights.shape[layer_ndim:]
        elif in_dpoly is not None and l_weights is not None:
            self.in_shape = in_dpoly.layer_shape
            layer_ndim = len(self.l_weights.shape) - len(self.in_shape)
            self.layer_shape = self.l_weights.shape[0:layer_ndim]
        elif l_bias is not None:
            self.layer_shape = np.array(l_bias).shape
        elif box.l is not None:
            self.layer_shape = box.l.shape
        else:
            raise Exception("Layer's shape can't be infered.")

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
        return np.hstack([np.expand_dims(self.l_bias, 1), self.l_weights])

    def u_combined(self) -> np.ndarray:
        return np.hstack([np.expand_dims(self.u_bias, 1), self.u_weights])

    def l_combined_ones(self) -> np.ndarray:
        l_combined = self.l_combined()
        return np.vstack([np.ones(l_combined.shape[1]), l_combined])

    def u_combined_ones(self) -> np.ndarray:
        u_combined = self.u_combined()
        return np.vstack([np.ones(u_combined.shape[1]), u_combined])

    def n_neur(self):
        """Returns the number of neurons in the layer."""
        return self.box.l.shape[0]

    def get_neur(self, idx):
        AbstractNeuron = namedtuple('AbstractNeuron', 'l u')
        return AbstractNeuron(
            l = self.l_combined()[idx],
            u = self.u_combined()[idx]
        )

    def layer_size(self):
        return np.product(self.layer_shape)
        
    def in_size(self):
        return np.product(self.in_shape)

    def update_neur(self, idx, new_lb, new_lw, new_ub, new_uw):
        self.l_bias[idx] = new_lb
        self.l_weights[idx] = new_lw
        self.u_bias[idx] = new_ub
        self.u_weights[idx] = new_uw

    def __repr__(self):
        lines = [f"DPoly {self.name}:"]
        for i in range(len(self.l_bias)):
            lines.append(f"neur{i}:")
            lines.append(f"\t>= {self.l_weights[i]} + {self.l_bias[i]}")
            lines.append(f"\t<= {self.u_weights[i]} + {self.u_bias[i]}")
            lines.append(f"\tl = {self.box.l[i]}")
            lines.append(f"\tu = {self.box.u[i]}")
        lines.append("\n")

        return "\n".join(lines)
        # "{0}\n{1}\n{2}\n{3}\nBox:\n{4}\n{5}\n\n"\
        #     .format(self.l_bias, self.l_weights, self.u_bias, self.u_weights, 
        #         self.box.l, self.box.u)
    