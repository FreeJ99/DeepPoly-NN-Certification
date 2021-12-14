import os
from verifier import main

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

gt = []
with open("../test_cases/gt.txt") as tc_f:
    lines = tc_f.readlines()

for line in lines:
    gt.append(line.split(',')[2].strip())

nets = ['fc1', 'fc2', 'fc3', 'fc4', 'fc5', 'fc6', 'fc7', 'conv1', 'conv2', 'conv3']

idx = 0
for net in nets:
    print(f"Evaluating {net}:")

    for spec in os.listdir(f"../test_cases/{net}"):
        out = main([
            "--net", net,
            "--spec", f"../test_cases/{net}/{spec}"
        ])
        if out == gt[idx]:
            print(f"{spec}\t{out}\t{bcolors.OKGREEN}OK{bcolors.ENDC}")
        else:
            print(f"{spec} \t{out}\t{bcolors.FAIL}WA{bcolors.ENDC}")
        print()
        idx += 1