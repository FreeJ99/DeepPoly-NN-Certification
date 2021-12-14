from typing import Tuple

import numpy as np
from box import Box

class DeepPoly():
    l_bias: np.ndarray
    l_weights: np.ndarray
    u_bias: np.ndarray
    u_weights: np.ndarray
    box: Box
    # in_dpoly: DeepPoly TODO 
    name: str # useful for debugging

    def __init__(self, in_dpoly, l_bias, l_weights, u_bias, u_weights, box, 
            name = ""):
        self.l_bias = l_bias
        self.l_weights = l_weights
        self.u_bias = u_bias
        self.u_weights = u_weights
        self.box = box
        self.in_dpoly = in_dpoly
        self.name = name
    
    def calculate_box(self):
        lW = self.l_weights
        lb = self.l_bias
        uW = self.u_weights
        ub = self.u_bias
        prev_box = self.in_dpoly.box
        box_l = lb + np.sum(np.minimum(lW * prev_box.l, lW * prev_box.u), 
            axis = 1)
        box_u = ub + np.sum(np.maximum(uW * prev_box.l, uW * prev_box.u), 
            axis = 1)
        self.box = Box(box_l, box_u)

    def l_combined(self) -> np.ndarray:
        return np.hstack([self.l_bias, self.l_weights])

    def u_combined(self) -> np.ndarray:
        return np.hstack([self.u_bias, self.u_weights])

    def l_combined_with_ones(self) -> np.ndarray:
        l_combined = self.l_combined()
        return np.vstack([np.ones(l_combined.shape[1]), l_combined])

    def u_combined_with_ones(self) -> np.ndarray:
        u_combined = self.u_combined()
        return np.vstack([np.ones(u_combined.shape[1]), u_combined])

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
    