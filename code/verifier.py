import argparse
import sys
import os

from numpy.core.defchararray import lower
import torch
from torch import nn
from networks import FullyConnected, Conv, Normalization
import numpy as np

DEVICE = 'cpu'
INPUT_SIZE = 28

def get_output_shape(layer: nn.Module, input_shape) -> int:
    if isinstance(layer, Normalization):
        return input_shape

    elif isinstance(layer, nn.Flatten):
        return (np.prod(input_shape), )

    elif isinstance(layer, nn.Linear):
        return (layer.out_features, )

    elif isinstance(layer, nn.ReLU):
        return input_shape

    else:
        return None

def get_box_bounds(layer: nn.Module, input_l: np.ndarray, input_u: np.ndarray) -> np.ndarray:
    # Box abstract transformers
    if isinstance(layer, Normalization):
        m = layer.mean.detach().numpy()
        s = layer.sigma.detach().numpy()
        lower_bounds = (input_l - m) / s
        upper_bounds = (input_u - m) / s

    elif isinstance(layer, nn.Flatten):
        lower_bounds = input_l.flatten()
        upper_bounds = input_u.flatten()

    elif isinstance(layer, nn.Linear):
        # check shapes
        W = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy()
        lower_bounds = b + np.sum(np.minimum(W * input_l, W * input_u), axis = 1)
        # print(f'Shape: {lower_bounds.shape}')
        upper_bounds = b + np.sum(np.maximum(W * input_l, W * input_u), axis = 1)

    elif isinstance(layer, nn.ReLU):
        lower_bounds = np.maximum(input_l, 0)
        upper_bounds = np.maximum(input_u, 0)

    else:
        return None

    return (lower_bounds, upper_bounds)

def analyze(net: nn.Module, inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    # Do a forward pass to calculate all ls and us
    # While the property can't be proven
    # Backsubstitute variables in existing inequlities
    # How I am going to represent these inequalities?
    net_layers = [module
        for module in net.modules()
        if not isinstance(module, (FullyConnected, nn.Sequential))]

    # Box abstract representation
    lower_bounds = {}
    upper_bounds = {}
    lower_bounds[inputs] = np.maximum(inputs.detach().numpy() - eps, 0)
    upper_bounds[inputs] = np.maximum(inputs.detach().numpy() + eps, 1)

    prev_layer = inputs
    for layer in net_layers:
        lower_bounds[layer], upper_bounds[layer] = \
            get_box_bounds(layer, lower_bounds[prev_layer], upper_bounds[prev_layer])
        prev_layer = layer
    
    print(lower_bounds[prev_layer])
    print(upper_bounds[prev_layer])

    final_l = lower_bounds[prev_layer]
    final_u = upper_bounds[prev_layer]
    target_l = final_l[true_label]
    other_idx = np.arange(len(final_u)) != true_label
    max_other_score = final_u[other_idx].max()

    if target_l > max_other_score:
        return True
    else:
        return False


def main(args):
    parser = argparse.ArgumentParser(description='Neural network verification using DeepZ relaxation')
    parser.add_argument('--net',
                        type=str,
                        choices=['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3'],
                        required=True,
                        help='Neural network architecture which is supposed to be verified.')
    parser.add_argument('--spec', type=str, required=True, help='Test case to verify.')
    args = parser.parse_args(args)

    print(os.getcwd())
    with open(args.spec, 'r') as f:
        lines = [line[:-1] for line in f.readlines()]
        true_label = int(lines[0])
        pixel_values = [float(line) for line in lines[1:]]
        eps = float(args.spec[:-4].split('/')[-1].split('_')[-1])

    if args.net == 'fc1':
        net = FullyConnected(DEVICE, INPUT_SIZE, [50, 10]).to(DEVICE)
    elif args.net == 'fc2':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 50, 10]).to(DEVICE)
    elif args.net == 'fc3':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 10]).to(DEVICE)
    elif args.net == 'fc4':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 50, 10]).to(DEVICE)
    elif args.net == 'fc5':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc6':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'fc7':
        net = FullyConnected(DEVICE, INPUT_SIZE, [100, 100, 100, 100, 100, 10]).to(DEVICE)
    elif args.net == 'conv1':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 3, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv2':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (32, 4, 2, 1)], [100, 10], 10).to(DEVICE)
    elif args.net == 'conv3':
        net = Conv(DEVICE, INPUT_SIZE, [(16, 4, 2, 1), (64, 4, 2, 1)], [100, 100, 10], 10).to(DEVICE)
    else:
        assert False

    net.load_state_dict(torch.load('../mnist_nets/%s.pt' % args.net, map_location=torch.device(DEVICE)))

    inputs = torch.FloatTensor(pixel_values).view(1, 1, INPUT_SIZE, INPUT_SIZE).to(DEVICE)
    outs = net(inputs)
    pred_label = outs.max(dim=1)[1].item()
    assert pred_label == true_label

    eps = 0.000001 # tmp
    if analyze(net, inputs, eps, true_label):
        # print('verified')
        return 'verified'
    else:
        # print('not verified')
        return 'not verified'


if __name__ == '__main__':
    print(main(sys.argv[1:]))
