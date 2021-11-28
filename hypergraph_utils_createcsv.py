import numpy as np
import scipy.sparse as sp
import user_side_hypergraphs_new
import pickle
import os


def create_timely_csv(args, interation, time, timestamp, time_split):
    all_user_item_interactions = {}
    to_folder = './HGCN/data/' + args.data + '/' + time_split + '/'
    if not os.path.exists(to_folder):
        os.makedirs(to_folder)
    subgraphs = {}

    subgraphs_mapping_u = {} # length t
    subgraphs_mapping_i = {} # length t
    for t in timestamp:
        subgraphs[t] = {'u':[], 'i':[]}
        subgraphs_mapping_u[t] = {} # for t, users
        subgraphs_mapping_i[t] = {} # for t, items
        all_user_item_interactions[t] = {}

    for user in interation:
        iteml = interation[user]
        timel = time[user]
        for i, t in zip(iteml, timel):
            if not i in subgraphs_mapping_i[t]:
                subgraphs_mapping_i[t][i] = len(subgraphs_mapping_i[t])
            if not user in subgraphs_mapping_u[t]:
                subgraphs_mapping_u[t][user] = len(subgraphs_mapping_u[t])

            subgraphs[t]['u'].append(subgraphs_mapping_u[t][user])
            subgraphs[t]['i'].append(subgraphs_mapping_i[t][i])


            if user not in all_user_item_interactions[t]:
                all_user_item_interactions[t][user] = []
            all_user_item_interactions[t][user].append(i)


    for t in timestamp:
        filename = to_folder + str(t) + '.csv'
        print(filename)
        f = open(filename, "w")
        for u in all_user_item_interactions[t]:
            i_list = all_user_item_interactions[t][u]

            for index, item in enumerate(all_user_item_interactions[t][u]):
                del i_list[:index]
                for tmp_item in i_list:
                    f.write(str(item)+','+str(tmp_item))
                    f.write('\n')
        f.close()
    exit(0)

def create_u_timely_csv(args, interation, time, timestamp):
    all_user_item_interactions = {}
    to_folder = './HGCN/data/' + args.data + '/userside/'
    subgraphs = {}

    subgraphs_mapping_u = {} # length t
    subgraphs_mapping_i = {} # length t
    for t in timestamp:
        subgraphs[t] = {'u':[], 'i':[]}
        subgraphs_mapping_u[t] = {} # for t, users
        subgraphs_mapping_i[t] = {} # for t, items
        all_user_item_interactions[t] = {}

    user_set = set()
    for user in interation:
        user_set.add(user)
        iteml = interation[user]
        timel = time[user]
        for i, t in zip(iteml, timel):
            if not i in subgraphs_mapping_i[t]:
                subgraphs_mapping_i[t][i] = len(subgraphs_mapping_i[t])
            if not user in subgraphs_mapping_u[t]:
                subgraphs_mapping_u[t][user] = len(subgraphs_mapping_u[t])

            subgraphs[t]['u'].append(subgraphs_mapping_u[t][user])
            subgraphs[t]['i'].append(subgraphs_mapping_i[t][i])


            if user not in all_user_item_interactions[t]:
                all_user_item_interactions[t][user] = []
            all_user_item_interactions[t][user].append(i)

    # all_u_similar_uids = user_side_hypergraphs_new.find_similar_users(list(timestamp), user_set, all_user_item_interactions)
    all_u_similar_uids = pickle.load(open('data/' + 'refine_' + args.data + '/' + 'similar_users.pkl', 'rb'))
    for t in timestamp:
        adj= {}
        filename = to_folder + str(t) + '.pkl'
        print(filename)
        for u in all_user_item_interactions[t]:
            try:
                if u not in all_u_similar_uids[t]:
                    continue
            except KeyError:
                break
            similar_users = all_u_similar_uids[t][u]
            similar_users.insert(0, u)
            u_side_interactions = {}
            for s_u in similar_users:
                if s_u not in u_side_interactions:
                    u_side_interactions[s_u] = all_user_item_interactions[t][s_u]

            item_pairs = []
            for u_side_u, u_side_is in u_side_interactions.items():
                i_list = u_side_is
                for index,item in enumerate(u_side_is):
                    del i_list[:index]
                    for tmp_item in i_list:
                        item_pair = str(item) + ',' + str(tmp_item)
                        item_pairs.append(item_pair)
            adj[u] = do_adj(item_pairs)

        to_file = open(to_folder+str(t)+'.pkl', "wb")
        pickle.dump(adj, to_file)
    exit(0)

def do_adj(item_pairs):
    object_to_idx = {}
    idx_counter = 0
    edges = []
    for line in item_pairs:
        n1, n2 = line.rstrip().split(',')
        if n1 in object_to_idx:
            i = object_to_idx[n1]
        else:
            i = idx_counter
            object_to_idx[n1] = i
            idx_counter += 1
        if n2 in object_to_idx:
            j = object_to_idx[n2]
        else:
            j = idx_counter
            object_to_idx[n2] = j
            idx_counter += 1
        edges.append((i, j))
    adj = np.zeros((len(object_to_idx), len(object_to_idx)))
    for i, j in edges:
        adj[i, j] = 1.  # comment this line for directed adjacency matrix
        adj[j, i] = 1.
    return adj