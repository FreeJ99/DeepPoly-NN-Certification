import os
import sys
import argparse
from time import time

from verifier import run_verifier


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


start = time()

gt = []
with open("test_cases/gt.txt") as tc_f:
    lines = tc_f.readlines()

for line in lines:
    gt.append(line.split(',')[2].strip())

nets = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5',
        'fc6', 'fc7']  # , 'conv1', 'conv2', 'conv3']

if __name__ == "__main__":
    print(sys.argv)
    parser = argparse.ArgumentParser(
        description='Evaluating robustness verifiers.')
    parser.add_argument('-v',
                        type=str,
                        choices=['box', 'deep_poly'],
                        help="Verifier to be used.",
                        required=True)
    args = parser.parse_args(sys.argv[1:])

    idx = 0
    for net in nets:
        print(f"Evaluating {net}:")
        it_start = time()
        for spec in os.listdir(f"test_cases/{net}"):
            result = run_verifier(net, f"test_cases/{net}/{spec}", args.v)
            if result == gt[idx]:
                print(f"{spec}\t{result}\t{bcolors.OKGREEN}OK{bcolors.ENDC}")
            else:
                print(f"{spec} \t{result}\t{bcolors.FAIL}WA{bcolors.ENDC}")
            print()
            idx += 1
        print(f'runtime: {time()-it_start}')

print(f'Total runtime {time()-start}')
