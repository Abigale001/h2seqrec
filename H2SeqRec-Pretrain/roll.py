import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num', default=20, type=int)
args = parser.parse_args()

print(random.randint(0, args.num))
