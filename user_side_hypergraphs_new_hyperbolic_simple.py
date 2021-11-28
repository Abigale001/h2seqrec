# -*- coding: utf-8 -*-
# @Time    : 2020/12/8 19:33
# @Author  : Yicong Li
# @FileName: user_side_hypergraphs.py
# @Software: PyCharm

import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import pandas as pd
from hypergraph_utils import generate_G_from_H
import pickle
import data.Dataset_monthly as Dataset

def subgraph_con(all_ui_interactions, all_u_similar_uids, existed_timestamp):
    subgraphs = {}
    subgraphs_G = {}

    subgraphs_mapping_u = {} # length t : {length u}
    subgraphs_mapping_i = {} # length t : {length u}

    for t in existed_timestamp:
        this_t_all_u_similar_uids = all_u_similar_uids[t]
        subgraphs[t] = {}
        subgraphs_mapping_u[t] = {}
        subgraphs_mapping_i[t] = {}
        for c_u in this_t_all_u_similar_uids:
            subgraphs[t][c_u] = {'u':[], 'i':[]}
            subgraphs_mapping_u[t][c_u] = {} # for t, users
            subgraphs_mapping_i[t][c_u] = {} # for t, items


    for t in existed_timestamp:
        this_t_all_u_similar_uids = all_u_similar_uids[t]
        this_t_all_ui_interactions = all_ui_interactions[t]
        # print(f'this_t_all_u_similar_uids:{this_t_all_u_similar_uids}')
        for c_u in this_t_all_u_similar_uids:
            # print(f'cu:{c_u}')
            # c_u: current user
            # similar_users: c_u's similar users list
            similar_users = this_t_all_u_similar_uids[c_u]
            similar_users.insert(0, c_u)
            small_u_side_interactions = {}
            for s_u in similar_users:
                if s_u not in small_u_side_interactions:
                    small_u_side_interactions[s_u] = this_t_all_ui_interactions[s_u]

            # print(f'small_u_side_interactions:{small_u_side_interactions}')
            for u, i_list in small_u_side_interactions.items():
                # print(u)
                # print(i_list)
                # print(subgraphs_mapping_i[t][c_u])
                for i in i_list:
                    if not i in subgraphs_mapping_i[t][c_u]:
                        subgraphs_mapping_i[t][c_u][i] = len(subgraphs_mapping_i[t][c_u])
                    if not u in subgraphs_mapping_u[t][c_u]:
                        subgraphs_mapping_u[t][c_u][u] = len(subgraphs_mapping_u[t][c_u])

                    subgraphs[t][c_u]['u'].append(subgraphs_mapping_u[t][c_u][u])
                    subgraphs[t][c_u]['i'].append(subgraphs_mapping_i[t][c_u][i])




    for t in existed_timestamp:
        this_t_all_u_similar_uids = all_u_similar_uids[t]
        this_t_all_ui_interactions = all_ui_interactions[t]
        subgraphs_G[t] = {}
        for c_u in this_t_all_u_similar_uids:
            col = subgraphs[t][c_u]['u']
            row = subgraphs[t][c_u]['i']
            data = np.ones(len(col))

            sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_i[t][c_u]), len(subgraphs_mapping_u[t][c_u])))
            print('Done constructing subgraph in timestamp', str(t))
            print(len(subgraphs_mapping_i[t][c_u]), len(subgraphs_mapping_u[t][c_u]), len(data))

            subgraphs_G[t][c_u] = {}
            # subgraphs_G[t]['G'], subgraphs_G[t]['E'] = generate_G_from_H(sg)
            subgraphs_G[t][c_u]['G'], subgraphs_G[t][c_u]['E'] = generate_G_from_H(sg) # 两层 for t, for u

    return subgraphs_mapping_i, subgraphs_G, subgraphs_mapping_u


def subgraph_key_building(subgraphs_mapping_i, users_list, num_items):
    reversed_subgraphs_mapping_i = {}
    # print(f'subgraphs_mapping_i: {subgraphs_mapping_i}')
    # print(f'len(subgraphs_mapping_i): {len(subgraphs_mapping_i)}')
    for t in subgraphs_mapping_i:
        reversed_subgraphs_mapping_i[t] = {}
        for c_u in subgraphs_mapping_i[t]:
            reversed_subgraphs_mapping_i[t][c_u] = [0] * len(subgraphs_mapping_i[t][c_u])
            for i in subgraphs_mapping_i[t][c_u]:
                reversed_subgraphs_mapping_i[t][c_u][subgraphs_mapping_i[t][c_u][i]] = i

    sorted_time = sorted(
        list(subgraphs_mapping_i.keys()))  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]



    return reversed_subgraphs_mapping_i, sorted_time


def find_similar_users(timestamps_l, users_list, all_u_i_interactions):
    """
    for each timestamp, find this user's similar users.
    What is a similar user?
        In each timestamp, if some u bought same thing, the user is similar with the current u.
    :param timestamps_l:
    :param current_uid:
    :param all_u_i_interactions:
    :return:
    """

    all_u_similar_uids = defaultdict()
    for t in timestamps_l:
        for current_uid in users_list:
            if current_uid in all_u_i_interactions[t]: # all_u_i_interactions[t]可能为空
                current_u_iids = all_u_i_interactions[t][current_uid]
                # all_u_similar_uids[t] = defaultdict()
                similar_users = []
                for uid, iids in all_u_i_interactions[t].items():

                    if_similar = False
                    # check if has same
                    for x in iids:
                        # traverse in the 2nd list
                        for y in current_u_iids:
                            # if one common
                            if x == y:
                                if_similar = True
                    if if_similar:
                        similar_users.append(uid)
                similar_users.remove(current_uid) # finally, get current user's similar users list
                if similar_users:
                    try:
                        all_u_similar_uids[t][current_uid] = similar_users
                    except KeyError:
                        all_u_similar_uids[t] = defaultdict(list)
                        all_u_similar_uids[t][current_uid] = similar_users

    return all_u_similar_uids


def create_user_side_hypergraphs(data, interation, time, timestamp):
    timestamp = sorted(timestamp, key=int)
    data_path = 'data/'
    path_to_data = data_path + 'refine_'+data + '/' + 'user_item.relation.newform'
    output_file = data_path + 'refine_'+data+'/'+'similar_users.pkl'


    all_ui_interactions = {key: defaultdict(defaultdict) for key in timestamp}
    # print(all_ui_interactions)

    users_set = set()

    for u_id, i_id_list in interation.items():
        users_set.add(u_id)
        times_u = time[u_id]
        for (i_id,t_id) in zip(i_id_list,times_u):
            if u_id not in all_ui_interactions[t_id]:
                all_ui_interactions[t_id][u_id] = []

            all_ui_interactions[t_id][u_id].append(i_id)






    all_u_similar_uids = pickle.load(open(output_file,'rb'))
    # print(all_u_similar_uids)
    existed_timestamp = list(all_u_similar_uids.keys())
    print(existed_timestamp)
    user_side_subgraphs_mapping_i, user_side_subgraphs_G, user_side_subgraphs_mapping_u = subgraph_con(all_ui_interactions, all_u_similar_uids, existed_timestamp)
    return user_side_subgraphs_mapping_i, user_side_subgraphs_G, user_side_subgraphs_mapping_u






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='AMT', help='data set name (default: newAmazon)')
    args = parser.parse_args()
    data = args.data
    train_set, val_set, train_val_set, test_set, data_time, neg_test, num_items, num_users = Dataset.data_partition_neg(
        args, 'month')
    output_file_1 = './data/refine_'+args.data+ '/'+'/user_side_subgraphs_mapping_i.pkl'
    output_file_2 = './data/refine_'+args.data+ '/'+'/user_side_subgraphs_G.pkl'
    output_file_3 = './data/refine_'+args.data+ '/'+'/user_side_subgraphs_mapping_u.pkl'

    user_side_subgraphs_mapping_i, user_side_subgraphs_G, user_side_subgraphs_mapping_u = create_user_side_hypergraphs(data,
        train_set, data_time[0], data_time[-2])
    pickle.dump(user_side_subgraphs_mapping_i, open(output_file_1, 'wb'))
    pickle.dump(user_side_subgraphs_G, open(output_file_2, 'wb'))
    pickle.dump(user_side_subgraphs_mapping_u, open(output_file_3, 'wb'))
    print(f'dump in {output_file_1}')
    print(f'dump in {output_file_2}')
    print(f'dump in {output_file_3}')
    # exit(0)
    # AMT
    num_items = 11673
    num_users = 5353

    # # Automotive
    # num_users = 3470
    # num_items = 5984

    # # newAmazon
    # num_users = 73731
    # num_items = 64061

    # # goodreads
    # num_users = 16702
    # num_items = 20824

    users_list = range(1,num_users)

    user_side_subgraphs_mapping_i = pickle.load(open(output_file_1,'rb'))
    user_side_subgraphs_G = pickle.load(open(output_file_2,'rb'))
    user_side_subgraphs_mapping_u = pickle.load(open(output_file_3,'rb'))

    reversed_subgraphs_mapping_i, sorted_time = subgraph_key_building(user_side_subgraphs_mapping_i, users_list, num_items)
    user_side_reversed_subgraphs_mapping_u, user_side_sorted_time_u = subgraph_key_building(
        user_side_subgraphs_mapping_u, users_list, num_users)

    output_file_4 = './data/refine_'+data+'/user_side_subgraphs_mapping_i2.pkl'
    output_file_5 = './data/refine_'+data+'/user_side_reversed_subgraphs_mapping_i.pkl'
    output_file_6 = './data/refine_'+data+'/user_side_sorted_time.pkl'
    output_file_7 = './data/refine_'+data+'/user_side_subgraphs_sequence_i.pkl'
    output_file_8 = './data/refine_'+data+'/user_side_subgraphs_mapping_u2.pkl'
    output_file_9 = './data/refine_'+data+'/user_side_reversed_subgraphs_mapping_u.pkl'
    output_file_10 = './data/refine_'+data+'/user_side_sorted_time_u.pkl'
    output_file_11 = './data/refine_'+data+'/user_side_subgraphs_sequence_u.pkl'

    pickle.dump(reversed_subgraphs_mapping_i, open(output_file_5, 'wb'))
    pickle.dump(sorted_time, open(output_file_6, 'wb'))
    pickle.dump(user_side_subgraphs_mapping_u, open(output_file_8, 'wb'))
    pickle.dump(user_side_reversed_subgraphs_mapping_u, open(output_file_9, 'wb'))
    pickle.dump(user_side_sorted_time_u, open(output_file_10, 'wb'))


    print('done')