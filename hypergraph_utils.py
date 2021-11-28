import numpy as np
import scipy.sparse as sp



def subgraph_con(interation, time, timestamp):
    subgraphs = {}
    subgraphs_G = {}

    subgraphs_mapping_u = {} # length t
    subgraphs_mapping_i = {} # length t
    for t in timestamp:
        subgraphs[t] = {'u':[], 'i':[]}
        subgraphs_mapping_u[t] = {} # for t, users
        subgraphs_mapping_i[t] = {} # for t, items

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


    for t in timestamp: 
        col = subgraphs[t]['u']
        row = subgraphs[t]['i']
        data = np.ones(len(col))

        sg = sp.coo_matrix((data, (row, col)), shape=(len(subgraphs_mapping_i[t]), len(subgraphs_mapping_u[t])))
        print('Done constructing subgraph', str(t))
        print(len(subgraphs_mapping_i[t]), len(subgraphs_mapping_u[t]), len(data))

        subgraphs_G[t] = {}
        subgraphs_G[t]['G'], subgraphs_G[t]['E'] = generate_G_from_H(sg)


    return subgraphs_mapping_i, subgraphs_G, subgraphs_mapping_u

def subgraph_key_building(subgraphs_mapping_i, num_items):

    reversed_subgraphs_mapping_i = {}
    # print(f'subgraphs_mapping_i: {subgraphs_mapping_i}')
    # print(f'len(subgraphs_mapping_i): {len(subgraphs_mapping_i)}')
    for t in subgraphs_mapping_i:
        reversed_subgraphs_mapping_i[t] = [0]*len(subgraphs_mapping_i[t])
        for i in subgraphs_mapping_i[t]:
            reversed_subgraphs_mapping_i[t][subgraphs_mapping_i[t][i]] = i
    
    sorted_time = sorted(list(subgraphs_mapping_i.keys())) #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    if not 0 in subgraphs_mapping_i:
        subgraphs_mapping_i[0] = {}
        for i in range(num_items):
            subgraphs_mapping_i[0][i] = i # why

    cumuindex = num_items
    cumuindex_record = {}
    for t in sorted_time:
        cumuindex_record[t] = cumuindex
        for i in subgraphs_mapping_i[t]:
            subgraphs_mapping_i[t][i] += cumuindex
        cumuindex += len(subgraphs_mapping_i[t]) # why

    ##### get the latest dynamic mapping the subgraph
    subgraphs_sequence_i = {}
    for i in range(1,num_items):
        subgraphs_sequence_i[i] = np.array([i] * (3 + len(sorted_time)))
        
    for t in sorted_time:
        for i in subgraphs_mapping_i[t]:
            try:
                subgraphs_sequence_i[i][t+1:] = subgraphs_mapping_i[t][i] # why # 前两个t空着。
            except KeyError:
                print(t)
                exit(0)

    reversed_subgraphs_mapping_last_i = {}
    for t in subgraphs_mapping_i:
        if t==0:
            continue
        reversed_subgraphs_mapping_last_i[t] = [0]*len(subgraphs_mapping_i[t])
        for i in subgraphs_mapping_i[t]:
            reversed_subgraphs_mapping_last_i[t][subgraphs_mapping_i[t][i]-cumuindex_record[t]] = subgraphs_sequence_i[i][t]
        
    return subgraphs_mapping_i, reversed_subgraphs_mapping_i, sorted_time, subgraphs_sequence_i, reversed_subgraphs_mapping_last_i



def generate_G_from_H(H):

    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.array(H.sum(1)) 
    # the degree of the hyperedge
    DE = np.array(H.sum(0))

    invDE2 = sp.diags(np.power(DE, -0.5).flatten())
    DV2 =  sp.diags(np.power(DV, -0.5).flatten())
    W = sp.diags(W)
    HT = H.T


    invDE_HT_DV2 = invDE2 * HT * DV2
    G = DV2 * H * W * invDE2 * invDE_HT_DV2
    return G, invDE_HT_DV2


def get_m_from_y_q_id_map(reversed_subgraphs_mapping_i_m, reversed_subgraphs_mapping_i_q, reversed_subgraphs_mapping_i_y):
    get_m_from_y_id_map = {}
    get_m_from_q_id_map = {}
    for t_m in reversed_subgraphs_mapping_i_m:
        get_m_from_y_id_map[t_m] = []
        get_m_from_q_id_map[t_m] = []

        # t_q = (t_m//4)+1
        t_q = ((t_m - 1) // 3)+1
        t_y = ((t_m - 1) // 12)+1

        y_list = reversed_subgraphs_mapping_i_y[t_y]
        q_list = reversed_subgraphs_mapping_i_q[t_q]
        m_list = reversed_subgraphs_mapping_i_m[t_m]


        for m_id in m_list:

            index_y = y_list.index(m_id)
            get_m_from_y_id_map[t_m].append(index_y)

            try:
                index_q = q_list.index(m_id)
            except ValueError:
                print(t_m, t_q, t_y)
            get_m_from_q_id_map[t_m].append(index_q)

    return get_m_from_y_id_map, get_m_from_q_id_map