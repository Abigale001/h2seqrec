import os
import math
import random
import pickle
import json
import csv
import time
from datetime import date, datetime
import collections
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')


def load_data(data_path):
    seqs = defaultdict(list)
    user_map, item_map = {}, {}
    date_map = defaultdict(list)
    user_set, item_set = set(), set()
    user_negs = defaultdict(list)
    # item_attr = {}

    file_name = data_path.strip("/") + "/user_item.relation.newform"
    with open(file_name, "r", encoding="utf8") as f:
        for row in f:
            row = row.strip("\n").split("\t")
            user_id, item_id = int(row[0]), int(row[1])
            date = str(datetime.strptime(row[-1], '%m %d, %Y').date())
            seqs[user_id].append(item_id)

            if user_id not in user_map:
                user_map[user_id] = defaultdict(list)
            if item_id not in item_map:
                item_map[item_id] = defaultdict(list)

            user_map[user_id][date].append(item_id)
            item_map[item_id][date].append(user_id)

            user_set.add(user_id)
            item_set.add(item_id)

    for uid in user_map.keys():
        for date, seq in user_map[uid].items():
            date_map[date].append((uid, seq))

    dates = [d for d in date_map.keys()]
    dates.sort()

    file_name = data_path.strip("/") + "/test_neg.txt.100.newform"
    skip_header = True
    with open(file_name, "r", encoding="utf8") as f:
        for row in f:
            if skip_header:
                skip_header = False
                continue
            row = row.strip("\n").split("\t")
            user_id, item_id = int(row[0]), int(row[1])
            user_negs[user_id].append(item_id)
            user_set.add(user_id)
            item_set.add(item_id)

    assert len(user_map) == len(user_set)
    assert len(item_map) == len(item_set)


    MAX_SEQ_LEN, MAX_USER_ID, MIN_SEQ_LEN = -1, -1, 9999
    for uid, seq in seqs.items():
        if uid > MAX_USER_ID:
            MAX_USER_ID = uid
        if len(seq) > MAX_SEQ_LEN:
            MAX_SEQ_LEN = len(seq)
        if len(seq) < MIN_SEQ_LEN:
            MIN_SEQ_LEN = len(seq)
    print("USERS:", len(user_map))
    print("MAX_USER_ID:", MAX_USER_ID)
    print("MAX_SEQ_LEN:", MAX_SEQ_LEN)
    print("MIN_SEQ_LEN:", MIN_SEQ_LEN)
    print()

    MAX_DATE_NUM, MIN_DATE_NUM = -1, 9999
    for date, seq in date_map.items():
        if len(seq) > MAX_DATE_NUM:
            MAX_DATE_NUM = len(seq)
        if len(seq) < MIN_DATE_NUM:
            MIN_DATE_NUM = len(seq)
    print("MAX_DATE_NUM:", MAX_DATE_NUM)
    print("MIN_DATE_NUM:", MIN_DATE_NUM)
    print()

    MAX_ITEM_ID = -1
    for item in item_set:
        if item > MAX_ITEM_ID:
            MAX_ITEM_ID = item
    print("ITMES:", len(item_set))
    print("MAX_ITEM_ID:", MAX_ITEM_ID)
    print()

    return seqs, user_map, user_negs, date_map, item_map, MAX_SEQ_LEN, MAX_ITEM_ID


def mask_sample(sequence):
    L = len(sequence) - 1
    pos_1, pos_2 = -1, -1
    for _ in range(L):
        pos_1 = random.randint(0, L - 1)
        for k in range(0, L):
            if pos_1 != k:
                return pos_1, k
    return -1, -1

##########################


def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT / len(pred_list), NDCG / len(pred_list), MRR / len(pred_list)


def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)


def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users


def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            sum_recall += len(act_set & pred_set) / float(len(act_set))
            true_users += 1
    return sum_recall / true_users


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def ndcg_k(actual, predicted, topk):
    res = 0
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
    return res / float(len(actual))


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res
