import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import sys
from tqdm import tqdm
import math
import random
from collections import defaultdict
import argparse
import logging
from time import time
import datetime
import pickle

from data import Dataset_monthly as Dataset
from interactions import Interactions
import hypergraph_utils as hgut
import user_side_hypergraphs as ushgut
import hypergraph_utils_createcsv as nhgut
from model_add_u import *
from sampler import *

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

parser = argparse.ArgumentParser()

# data arguments
parser.add_argument('--seq_len', type=int, default=50, help='max sequence length (default: 20)')
parser.add_argument('--data', type=str, default='goodreads', help='data set name (default: newAmazon)')
parser.add_argument('--T', type=int, default=1)

# use BPR to pretrain
parser.add_argument('--n_iter_bpr', type=int, default=20)
parser.add_argument('--bpr_batch_size', type=int, default=512)
parser.add_argument('--worker_bpr', type=int, default=2, help='number of sampling workers (default: 2)')

parser.add_argument('--n_iter', type=int, default=200)
parser.add_argument('--seed', type=int, default=1111)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--l2_reg', type=float, default=0.0)
parser.add_argument('--clip', type=float, default=1., help='gradient clip (default: 1.)')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout (default: 0.5)')
parser.add_argument('--graph_dropout', type=float, default=0.6, help='dropout (default: 0.6)')
parser.add_argument('--eval_interval', type=int, default=10, help='eval/test interval')
parser.add_argument('--worker', type=int, default=5, help='number of sampling workers (default: 10)')
parser.add_argument('--eval_batch_size', type=int, default=1024, help='eval/test batch size (default: 128)')

parser.add_argument('--emsize', type=int, default=100, help='dimension of item embedding (default: 100)')
parser.add_argument('--n_hyper', type=int, default=2, help='number of layers for Hypergraph (default: 2)')
parser.add_argument('--num_blocks', type=int, default=2, help='num_blocks')
parser.add_argument('--num_heads', type=int, default=1, help='num_heads')
parser.add_argument('--pos_fixed', type=int, default=0, help='trainable positional embedding usually has better performance')

args = parser.parse_args()
tf.set_random_seed(args.seed)

data = args.data
args.neg_size = 1

train_set, val_set, train_val_set, test_set, data_time, neg_test, num_items, num_users = Dataset.data_partition_neg(args, 'month')


# print(data_time[0])
# print(data_time[-1])
# print(data_time[-2])
print(f'num_items:{num_items}\tnum_users:{num_users}')

nhgut.create_u_timely_csv(args, train_set, data_time[0], data_time[-2])