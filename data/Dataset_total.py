#encoding=utf-8
import pickle
import math

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from collections import defaultdict
import datetime
import operator

data_path = 'data/'
#same with dataset_hie

def data_partition_neg(args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_time_m = defaultdict(list)
    User_time_q = defaultdict(list)
    User_time_y = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_train_valid = {}
    user_test = {}
    neg_test = {}
    all_ui_interactions = defaultdict(defaultdict)

    time_set_train_m = set()
    time_set_train_q = set()
    time_set_train_y = set()
    time_set_test_m = set()
    time_set_test_q = set()
    time_set_test_y = set()

    user_train_time_m = {}
    user_train_time_q = {}
    user_train_time_y = {}

    user_valid_time_m = {}
    user_valid_time_q = {}
    user_valid_time_y = {}
    user_train_valid_time_m = {}
    user_train_valid_time_q = {}
    user_train_valid_time_y = {}
    user_test_time_m = {}
    user_test_time_q = {}
    user_test_time_y = {}
    # assume user/item index starting from 1
    # path_to_data = data_path + args.data + '/' + args.data + '_all.txt'
    path_to_data = data_path + 'refine_'+args.data + '/' + 'user_item.relation.newform'
    # path_to_data = data_path + 'refine_AM/' + 'train.csv'

    # 根据不同的数据集进行设计，指示不同的年份，以及上下半年
    # for Amazon    
    # t_map = {1997:[1], 1998:[1], 1999:[1], 2000:[1], 2001:[1], 2002:[1], 2003:[1], 2004:[1], 2005:[1], 2006:[1], 2007:[1], 2008:[1], 2009:[2,3], 2010:[4,5], \
    # 2011:[6,7], 2012:[8,9], 2013:[10,11], 2014:[12,13], 2015:[14,15], 2016:[16,17], 2017:[18,19], 2018:[20]}

    # how to hierarchical build hypergraphs
    # monthly
    if args.data == 'AMT' or args.data == 'newAmazon':
        t_map_m = {2014:[1,2,3,4,5,6,7,8,9,10,11,12],
                 2015:[13,14,15,16,17,18,19,20,21,22,23,24],
                 2016:[25,26,27,28,29,30,31,32,33,34,35,36],
                 2017:[37,38,39,40,41,42,43,44,45,46,47,48],
                 2018:[49]}

        # quarterly
        t_map_q = {2014:[1,2,3,4],
                 2015:[5,6,7,8],
                 2016:[9,10,11,12],
                 2017:[13,14,15,16],
                 2018:[17]}

        # yearly
        t_map_y = {2014:1, 2015:2, 2016:3, 2017:4, 2018:5}
    if args.data == 'goodreads':
        t_map_m = {2013: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                 2014: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],
                 2015: [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]}

        # quarterly
        t_map_q = {2013: [1, 2, 3, 4],
                 2014: [5, 6, 7, 8],
                 2015: [9, 10, 11, 12]}

        # yearly
        t_map_y = {2013: 1, 2014: 2, 2015: 3}

    # 指示上下半年
    # m_map = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:1, 8:1, 9:1, 10:1, 11:1, 12:1}


    f = open(path_to_data, 'r')
    for line in f:
        u, i, t, d = line.rstrip().split('\t')
        u = int(u)
        i = int(i)
        # year = int(datetime.datetime.fromtimestamp(int(t)).strftime("%Y")) # Day of the year as a decimal number [001,366]
        # month = int(datetime.datetime.fromtimestamp(int(t)).strftime("%m"))

        year = int(d.split(',')[-1])
        month = int(d.split(' ')[0])

        usernum = max(u, usernum)
        itemnum = max(i, itemnum)
        User[u].append(i)   

        temp_map_m = t_map_m[year]
        temp_map_q = t_map_q[year]
        temp_map_y = t_map_y[year]

        # monthly
        if year == 2018:
            User_time_m[u].append(temp_map_m[0])
        else:
            User_time_m[u].append(temp_map_m[month - 1])

        # quarterly
        if year == 2018:
            User_time_q[u].append(temp_map_q[0])
        else:
            User_time_q[u].append(temp_map_q[int((month-1)//3)])

        # yearly
        User_time_y[u].append(t_map_y[year])

    # print(f'User_time: {User_time}')

    for user in User: # u-i dict
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            user_train_time_m[user] = User_time_m[user]
            user_train_time_q[user] = User_time_q[user]
            user_train_time_y[user] = User_time_y[user]
            user_valid_time_m[user] = []
            user_valid_time_q[user] = []
            user_valid_time_y[user] = []
            user_test_time_m[user] = []
            user_test_time_q[user] = []
            user_test_time_y[user] = []
        else:
            user_train[user] = User[user][:-2] # train: delete last 2 from User
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # valid: last 2
            user_test[user] = []
            user_test[user].append(User[user][-1]) # test last 1
            
            neg_test[user] = [User[user][-1]] # test last 1 : positive

            user_train_time_m[user] = User_time_m[user][:-2] # user:time_list
            user_train_time_q[user] = User_time_q[user][:-2]
            user_train_time_y[user] = User_time_y[user][:-2]
            try:
                time_set_train_m.update(user_train_time_m[user]) # 涉及的time set
                time_set_train_q.update(user_train_time_q[user])
                time_set_train_y.update(user_train_time_y[user])
            except TypeError:
                print(user_train_time_m[user])
                print(user_train_time_q[user])
                print(user_train_time_y[user])
            user_valid_time_m[user] = []
            user_valid_time_q[user] = []
            user_valid_time_y[user] = []
            user_valid_time_m[user].append(User_time_m[user][-2]) # valid: user -> time
            user_valid_time_q[user].append(User_time_q[user][-2]) # valid: user -> time
            user_valid_time_y[user].append(User_time_y[user][-2]) # valid: user -> time

            user_test_time_m[user] = []
            user_test_time_q[user] = []
            user_test_time_y[user] = []

            user_test_time_m[user].append(User_time_m[user][-1]) # test: user -> time
            user_test_time_q[user].append(User_time_q[user][-1]) # test: user -> time
            user_test_time_y[user].append(User_time_y[user][-1]) # test: user -> time

            time_set_test_m.update(user_test_time_m[user])
            time_set_test_q.update(user_test_time_q[user])
            time_set_test_y.update(user_test_time_y[user])



        user_train_valid[user] = user_train[user] + user_valid[user] # train+valid: user -> item
        user_train_valid_time_m[user] = user_train_time_m[user] + user_valid_time_m[user] # train+valid: user -> time
        user_train_valid_time_q[user] = user_train_time_q[user] + user_valid_time_q[user] # train+valid: user -> time
        user_train_valid_time_y[user] = user_train_time_y[user] + user_valid_time_y[user] # train+valid: user -> time



    skip = 0
    # neg_f = data_path + args.data + '_test_neg.txt'
    neg_f = data_path + 'refine_' + args.data + '/' + 'test_neg.txt.'+str(args.test_neg_size)+'.newform'
    # neg_f = data_path + 'refine_AM/neg.csv'
    with open(neg_f, 'r') as file:
        for line in file:
            skip += 1
            if skip==1:
                continue
            user_id, item_id = line.rstrip().split('\t')
            u = int(user_id)
            i = int(item_id)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)

            neg_test[u].append(i)

    sequences = np.zeros((usernum + 1, args.test_neg_size+1),dtype=np.int64)
    for user in User:
        sequences[user][:] = neg_test[user]

    neg_test = sequences.copy()

    return [user_train, user_valid, user_train_valid, user_test, (user_train_time_m, user_train_time_q, user_train_time_y,
                                                                  user_valid_time_m, user_valid_time_q, user_valid_time_y,
                                                                  user_train_valid_time_m ,user_train_valid_time_q,user_train_valid_time_y,
                                                                  user_test_time_m,user_test_time_q, user_test_time_y,
                                                                  time_set_train_m,time_set_train_q, time_set_train_y,
                                                                  time_set_test_m, time_set_test_q, time_set_test_y),
            neg_test, itemnum+1, usernum+1]
