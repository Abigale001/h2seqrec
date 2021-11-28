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

def data_partition_neg(args, time_split):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    User_time = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_train_valid = {}
    user_test = {}
    neg_test = {}
    all_ui_interactions = defaultdict(defaultdict)

    time_set_train = set()
    time_set_test = set()

    user_train_time = {}
    user_valid_time = {}
    user_train_valid_time = {}
    user_test_time = {}
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
    # newAmazon; AMT
    if args.data == 'AMT' or args.data == 'newAmazon':
        if time_split == 'month':
            t_map = {2014:[1,2,3,4,5,6,7,8,9,10,11,12],
                     2015:[13,14,15,16,17,18,19,20,21,22,23,24],
                     2016:[25,26,27,28,29,30,31,32,33,34,35,36],
                     2017:[37,38,39,40,41,42,43,44,45,46,47,48],
                     2018:[49]}
        elif time_split == 'quarter':
            t_map = {2014:[1,2,3,4],
                     2015:[5,6,7,8],
                     2016:[9,10,11,12],
                     2017:[13,14,15,16],
                     2018:[17]}
        elif time_split == 'year':
            t_map = {2014: 1, 2015: 2, 2016: 3, 2017: 4, 2018: 5}

    if args.data == 'goodreads':
        if time_split == 'month':
            t_map = {2013: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     2014:[13,14,15,16,17,18,19,20,21,22,23,24],
                     2015:[25,26,27,28,29,30,31,32,33,34,35,36]}
        elif time_split == 'quarter':
            t_map = {2013:[1,2,3,4],
                    2014:[5,6,7,8],
                     2015:[9,10,11,12]}
        elif time_split == 'year':
            t_map = {2013:1, 2014: 2, 2015: 3}


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

        temp_map = t_map[year]

        # monthly
        # newAmazon&AMT
        if time_split == 'month':
            if year == 2018:
                User_time[u].append(temp_map[0])
            else:
                User_time[u].append(temp_map[month - 1])

        # quarterly
        if time_split == 'quarter':
            if year == 2018:
                User_time[u].append(temp_map[0])
            else:
                User_time[u].append(t_map[year][int((month-1)/3)])

        # yearly
        if time_split == 'year':
            User_time[u].append(t_map[year])

    # print(f'User_time: {User_time}')

    for user in User: # u-i dict
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []

            user_train_time[user] = User_time[user]
            user_valid_time[user] = []
            user_test_time[user] = []
        else:
            user_train[user] = User[user][:-2] # train: delete last 2 from User
            user_valid[user] = []
            user_valid[user].append(User[user][-2]) # valid: last 2
            user_test[user] = []
            user_test[user].append(User[user][-1]) # test last 1
            
            neg_test[user] = [User[user][-1]] # test last 1 : positive

            user_train_time[user] = User_time[user][:-2] # user:time_list
            time_set_train.update(user_train_time[user]) # 涉及的time set
            user_valid_time[user] = []
            user_valid_time[user].append(User_time[user][-2]) # valid: user -> time
            user_test_time[user] = []
            user_test_time[user].append(User_time[user][-1]) # test: user -> time
            time_set_test.update(user_test_time[user])


        user_train_valid[user] = user_train[user] + user_valid[user] # train+valid: user -> item
        user_train_valid_time[user] = user_train_time[user] + user_valid_time[user] # train+valid: user -> time


    skip = 0
    # neg_f = data_path + args.data + '_test_neg.txt'
    # neg_f = data_path + 'refine_' + args.data + '/' + 'test_neg.txt.100.newform'
    # neg_f = data_path + 'refine_' + args.data + '/' + 'test_neg.txt.500.newform'
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

    return [user_train, user_valid, user_train_valid, user_test, (user_train_time, user_valid_time, \
        user_train_valid_time, user_test_time, time_set_train, time_set_test), neg_test, itemnum+1, usernum+1]
