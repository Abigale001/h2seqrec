#encoding=utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import sys
import scipy.sparse as sp

from base import TransformerNet
from layers import HGNN_conv
import pickle
from layers import NN

def log2(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape


class NeuralSeqRecommender(object):
    def __init__(self, args, n_items, n_users,
                 subgraphs_G_m, reversed_subgraphs_mapping_i_m, reversed_subgraphs_mapping_u_m,
                 reversed_subgraphs_mapping_last_i_m, sorted_time_m,
                 subgraphs_G_q, reversed_subgraphs_mapping_i_q, reversed_subgraphs_mapping_u_q,
                 reversed_subgraphs_mapping_last_i_q, sorted_time_q,
                 subgraphs_G_y, reversed_subgraphs_mapping_i_y, reversed_subgraphs_mapping_u_y,
                 reversed_subgraphs_mapping_last_i_y, sorted_time_y,
                 get_m_from_y_id_map, get_m_from_q_id_map,
                 subgraphs_G_group, reversed_subgraphs_mapping_i_group, reversed_subgraphs_mapping_u_group,
                 user_side_sorted_time,
                 n_hyper = 2):
        self.args = args
        # add_pretrain = ['initial', 'fuse', '']
        self.add_pretrain = args.add_pretrain
        self.n_items = n_items
        self.n_users = n_users
        self.sorted_time_m = sorted_time_m
        self.sorted_time_q = sorted_time_q
        self.sorted_time_y = sorted_time_y

        self.user_side_sorted_time = user_side_sorted_time

        self.n_hyper = n_hyper      #number of hypergraph layers

        self.reversed_subgraphs_mapping_i_m = reversed_subgraphs_mapping_i_m   #for the original self-attention model
        self.reversed_subgraphs_mapping_i_q = reversed_subgraphs_mapping_i_q   #for the original self-attention model
        self.reversed_subgraphs_mapping_i_y = reversed_subgraphs_mapping_i_y   #for the original self-attention model

        self.reversed_subgraphs_mapping_u_m = reversed_subgraphs_mapping_u_m
        self.reversed_subgraphs_mapping_u_q = reversed_subgraphs_mapping_u_q
        self.reversed_subgraphs_mapping_u_y = reversed_subgraphs_mapping_u_y

        self.reversed_subgraphs_mapping_last_i_m = reversed_subgraphs_mapping_last_i_m
        self.reversed_subgraphs_mapping_last_i_q = reversed_subgraphs_mapping_last_i_q
        self.reversed_subgraphs_mapping_last_i_y = reversed_subgraphs_mapping_last_i_y

        # self.subgraphs_G_group = {}
        self.reversed_subgraphs_mapping_i_group = reversed_subgraphs_mapping_i_group

        self.subgraphs_G_m = {}
        self.subgraphs_G_q = {}
        self.subgraphs_G_y = {}

        self.get_m_from_y_id_map = get_m_from_y_id_map
        self.get_m_from_q_id_map = get_m_from_q_id_map

        self.test_neg_size = args.test_neg_size

        for i in sorted_time_m:
            self.subgraphs_G_m[i] = {}
            self.subgraphs_G_m[i]['G'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_m[i]['G'])
            self.subgraphs_G_m[i]['E'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_m[i]['E'])

        for i in sorted_time_q:
            self.subgraphs_G_q[i] = {}
            self.subgraphs_G_q[i]['G'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_q[i]['G'])
            self.subgraphs_G_q[i]['E'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_q[i]['E'])

        for i in sorted_time_y:
            self.subgraphs_G_y[i] = {}
            self.subgraphs_G_y[i]['G'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_y[i]['G'])
            self.subgraphs_G_y[i]['E'] = _convert_sp_mat_to_sp_tensor(subgraphs_G_y[i]['E'])


        self._build()
        self.build_bpr_graph()

        self.saver = tf.train.Saver()




    def build_bpr_graph(self):
        self.triple_bpr = tf.placeholder(dtype=tf.int32, shape=[None, 3])
        with tf.name_scope('con_bpr'):
            x_pos, x_neg = self.infer_bpr(self.triple_bpr)
            self.loss_bpr = tf.reduce_sum(tf.log(1 + tf.exp(-(x_pos - x_neg))))
        with tf.name_scope('training'):
            self.optimizer_bpr = tf.train.AdagradOptimizer(learning_rate=self.lr).minimize(self.loss_bpr)


    def infer_bpr(self, triple_bpr):
        with tf.name_scope('lookup_bpr'):
            bpr_user = tf.nn.embedding_lookup(self.user_embedding, triple_bpr[:, 0])
            bpr_pos = tf.nn.embedding_lookup(self.item_embedding, triple_bpr[:, 1])
            bpr_neg = tf.nn.embedding_lookup(self.item_embedding, triple_bpr[:, 2])

        with tf.name_scope('cal_bpr'):
            x_pos = tf.reduce_sum(tf.multiply(bpr_user, bpr_pos), axis=1)
            x_neg = tf.reduce_sum(tf.multiply(bpr_user, bpr_neg), axis=1)

        return x_pos, x_neg


    def _init_weights(self):
    	self.all_weights = {}

    	initializer = tf.contrib.layers.xavier_initializer()

    	for i in self.sorted_time:
    		for n in range(self.n_hyper):
        		self.all_weights['W_'+str(i)+'_'+str(n)] = tf.Variable(initializer([self.args.emsize, self.args.emsize]), name='W_'+str(i)+'_'+str(n))

    def _build(self):

        self.inp = tf.placeholder(tf.int32, shape=(None, None), name='inp')
        self.inp_ori = tf.placeholder(tf.int32, shape=(None, None), name='inp_ori')
        self.inp_user = tf.placeholder(tf.int32, shape=(None, None), name='inp_user')
        self.pos = tf.placeholder(tf.int32, shape=(None, None), name='pos')
        self.neg = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg')
        self.pos_dy = tf.placeholder(tf.int32, shape=(None, None), name='pos_dy')
        self.neg_dy = tf.placeholder(tf.int32, shape=(None, self.args.neg_size), name='neg_dy')

        self.u_list = tf.placeholder(tf.int32, shape=(None), name='u_list')
        self.u_list_dy = tf.placeholder(tf.int32, shape=(None), name='u_list_dy')

        self.u_seq = tf.placeholder(tf.int32, shape=(None, None), name='u_seq')


        self.lr = tf.placeholder(tf.float32, shape=None, name='lr')
        self.dropout = tf.placeholder_with_default(0., shape=())
        self.dropout_graph = tf.placeholder_with_default(0., shape=())
        self.item_embedding = item_embedding = tf.get_variable('item_embedding', \
                                shape=(self.n_items, self.args.emsize), \
                                dtype=tf.float32, \
                                regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                                initializer=tf.contrib.layers.xavier_initializer())
        

        self.user_embedding = user_embedding =  tf.get_variable('user_embedding', \
                        shape=(self.n_users, self.args.emsize), \
                        dtype=tf.float32, \
                        regularizer=tf.contrib.layers.l2_regularizer(self.args.l2_reg), \
                        initializer=tf.contrib.layers.xavier_initializer())

        self.item_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                      item_embedding[1:, :]), 0)

        

        self.user_embedding = tf.concat((tf.zeros(shape=[1, self.args.emsize]),
                                      user_embedding[1:, :]), 0)


        emb_list_m = self.item_embedding
        emb_list_q = self.item_embedding
        emb_list_y = self.item_embedding
        item_embs = self.item_embedding
        emb_m = []
        emb_q = []
        emb_y = []
        emb_list = self.item_embedding
        emb_list_user = [self.user_embedding]

        # for i in self.sorted_time_m:
        #     x1 = tf.nn.embedding_lookup(self.item_embedding, self.reversed_subgraphs_mapping_i_m[i])
        #     x2 = tf.nn.embedding_lookup(emb_list_m, self.reversed_subgraphs_mapping_last_i_m[i])
        #
        #     stacked_features = tf.stack([x1,x2])
        #
        #
        #     stacked_features_transformed = tf.layers.dense(stacked_features, self.args.emsize, activation=tf.nn.tanh)
        #     stacked_features_score = tf.layers.dense(stacked_features_transformed, 1)
        #     stacked_features_score = tf.nn.softmax(stacked_features_score, 0)
        #     stacked_features_score = tf.nn.dropout(stacked_features_score, keep_prob=1. - self.dropout)
        #
        #     xx = tf.reduce_sum(stacked_features_score*stacked_features, 0)
        #
        #
        #     nodes, edges = HGNN_conv(input_dim = self.args.emsize,
        #                                    output_dim = self.args.emsize,
        #                                    adj = self.subgraphs_G_m[i],
        #                                    act = tf.nn.relu,
        #                                    dropout = self.dropout_graph,
        #                                    n_hyper = self.n_hyper)(xx)
        #
        #     # emb_list_m = tf.concat([emb_list_m, nodes],0)
        #     emb_m.append(nodes)
        #     emb_list_user.append(edges)

        if self.add_pretrain == 'initial':
            if self.args.data == 'goodreads':
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/Hyperboloid/euc/'
            else:
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/euc/'
        else:
            path = 'data/refine_' + self.args.data + '/month/euc/'

        for i in self.sorted_time_m:
            npy_path = path + str(i) + '.euc.npy'
            load_embs = np.load(npy_path)
            nodes = tf.convert_to_tensor(load_embs, dtype=tf.float32)

            edges = NN(input_dim=self.args.emsize,
                       output_dim=self.args.emsize,
                       adj=self.subgraphs_G_m[i],
                       act=tf.nn.relu,
                       dropout=self.dropout_graph,
                       n_hyper=self.n_hyper)(nodes)
            emb_m.append(nodes)
            emb_list_m = tf.concat([emb_list_m, nodes], 0)
            emb_list_user.append(edges)

        # len(emb_list_user) = 50 # 0: (5353,100), 1:(200,100), 2: (180,100)...
        emb_list_user = tf.concat(emb_list_user, 0) # 30572,100

        if self.add_pretrain == 'initial':
            if self.args.data == 'goodreads':
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/Hyperboloid/euc/'
            else:
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/euc/'
        else:
            path = 'data/refine_' + self.args.data + '/year/euc/'
        for i in self.sorted_time_y:
            npy_path = path + str(i) + '.euc.npy'
            load_embs = np.load(npy_path)
            nodes = tf.convert_to_tensor(load_embs, dtype=tf.float32)

            emb_y.append(nodes)

        if self.add_pretrain == 'initial':
            if self.args.data == 'goodreads':
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/Hyperboloid/euc/'
            else:
                path = 'data/refine_' + self.args.data + '/month/use_pretrain/euc/'
        else:
            path = 'data/refine_' + self.args.data + '/quarter/euc/'
        for i in self.sorted_time_q:
            npy_path = path + str(i) + '.euc.npy'
            load_embs = np.load(npy_path)
            nodes = tf.convert_to_tensor(load_embs, dtype=tf.float32)

            emb_q.append(nodes)

        # print(emb_list_m.shape)
        # print(emb_list_q.shape)
        # print(emb_list_y.shape)

        for month_id, monthly_item_emb in enumerate(emb_m):
            quauter_id = ((month_id) // 3)
            year_id = ((month_id) // 12)

            year_item_emb = emb_y[year_id]
            year_month = tf.nn.embedding_lookup(year_item_emb, self.get_m_from_y_id_map[month_id+1])
            stacked_features = tf.stack([monthly_item_emb, year_month])
            stacked_features_transformed_item_embs = tf.layers.dense(stacked_features, self.args.emsize,
                                                                     activation=tf.nn.tanh)
            stacked_features_score_item_embs = tf.layers.dense(stacked_features_transformed_item_embs, 1)
            stacked_features_score_item_embs = tf.nn.softmax(stacked_features_score_item_embs, 0)
            stacked_features_score_item_embs = tf.nn.dropout(stacked_features_score_item_embs,
                                                             keep_prob=1. - self.dropout)
            mix_month_year_item_embs = tf.reduce_sum(stacked_features_score_item_embs * stacked_features, 0)

            quarter_item_emb = emb_q[quauter_id]
            quarter_month = tf.nn.embedding_lookup(quarter_item_emb, self.get_m_from_q_id_map[month_id+1])
            stacked_features = tf.stack([mix_month_year_item_embs, quarter_month])
            stacked_features_transformed_item_embs = tf.layers.dense(stacked_features, self.args.emsize,
                                                                     activation=tf.nn.tanh)
            stacked_features_score_item_embs = tf.layers.dense(stacked_features_transformed_item_embs, 1)
            stacked_features_score_item_embs = tf.nn.softmax(stacked_features_score_item_embs, 0)
            stacked_features_score_item_embs = tf.nn.dropout(stacked_features_score_item_embs,
                                                             keep_prob=1. - self.dropout)
            mix_month_quarter_item_embs = tf.reduce_sum(stacked_features_score_item_embs * stacked_features, 0)

            item_embs = tf.concat([item_embs, mix_month_quarter_item_embs],0)




        input_item1 = tf.nn.embedding_lookup(item_embs, self.inp) # dynamic item embeddings
        input_item1 = input_item1 * (self.args.emsize ** 0.5)


        input_user = tf.nn.embedding_lookup(emb_list_user, self.u_seq)
        input_user = input_user * (self.args.emsize ** 0.5)


        input_item2 = tf.nn.embedding_lookup(self.item_embedding, self.inp_ori)
        input_item2 = input_item2 * (self.args.emsize ** 0.5)

        try:
            # user_side item learning
            item_emb_group = [tf.zeros(shape=[1, self.args.emsize])]*(self.n_items)
            path = 'data/refine_'+self.args.data+'/userside/euc/'
            for t in self.user_side_sorted_time:

                pickle_dict = path + str(t) +'.pkl'
                try:
                    u_items_dict = pickle.load(open(pickle_dict, 'rb'))
                except IOError:
                    print(f'{pickle_dict} IOError')
                    continue

                for u in u_items_dict:
                    item_load = u_items_dict[u]
                    nodes = tf.convert_to_tensor(item_load, dtype=tf.float32)

                    static_items = tf.nn.embedding_lookup(self.item_embedding, self.reversed_subgraphs_mapping_i_group[t][u])


                    stacked_features_group = tf.stack([static_items, nodes])  # static and dynamic item embeddings

                    stacked_features_transformed_group = tf.layers.dense(stacked_features_group, self.args.emsize,
                                                                   activation=tf.nn.tanh)
                    stacked_features_score_group = tf.layers.dense(stacked_features_transformed_group, 1)
                    stacked_features_score_group = tf.nn.softmax(stacked_features_score_group, 0)
                    stacked_features_score_group = tf.nn.dropout(stacked_features_score_group, keep_prob=1. - self.dropout)

                    updated_items_group = tf.reduce_sum(stacked_features_score_group * stacked_features_group, 0)
                    for i, item_index in enumerate(self.reversed_subgraphs_mapping_i_group[t][u]):

                        item_emb_group[item_index] = tf.squeeze(updated_items_group[i])
                        # print(item_emb_group[item_index].shape)
                    for index, item_emb in enumerate(item_emb_group):
                        item_emb_group[index] = tf.squeeze(item_emb_group[index])
            input_item3 = tf.convert_to_tensor(item_emb_group, dtype=tf.float32)

            stacked_features_input = tf.stack([input_item1, input_item2, input_item3, input_user])
        except ValueError:

            print('Jump user-side..........n\n\n\n\n\n\n\\n\n\n\n\n\n\n\n')
            stacked_features_input = tf.stack([input_item1, input_item2, input_user])



        if self.add_pretrain == 'fuse':
            dataset_base = 'data/' + 'refine_' + self.args.data + '/'
            if self.args.data == 'AMT':
                pre_feature = pickle.load(open(dataset_base + '/use_pretrain/pretrain-AMT.2021.01.25.12.07.37.pkl', 'rb'))
            elif self.args.data == 'goodreads':
                pre_feature = pickle.load(
                    open(dataset_base + '/use_pretrain/pretrain-goodreads.2021.01.25.dim.100.pkl', 'rb'))
            input_item_pre = tf.nn.embedding_lookup(pre_feature, self.inp_ori)
            input_item_pre = tf.cast(tf.expand_dims(input_item_pre, 0), tf.float32)
            stacked_features_input = tf.concat([stacked_features_input, input_item_pre], axis=0)

        # fuse them
        stacked_features_transformed_input = tf.layers.dense(stacked_features_input, self.args.emsize, activation=tf.nn.tanh)
        stacked_features_score_input = tf.layers.dense(stacked_features_transformed_input, 1)
        stacked_features_score_input = tf.nn.softmax(stacked_features_score_input, 0)
        stacked_features_score_input = tf.nn.dropout(stacked_features_score_input, keep_prob=1. - self.dropout)

        input_item_all = tf.reduce_sum(stacked_features_score_input*stacked_features_input, 0)



        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inp_ori, 0)), -1)

        self.net = TransformerNet(self.args.emsize, self.args.num_blocks, self.args.num_heads, self.args.seq_len, dropout_rate=self.dropout, pos_fixed=self.args.pos_fixed,reuse=True)


        outputs = self.net(input_item_all, mask)

        

        ct_vec_last = outputs[:,-1,:]
        ct_vec_last = tf.reshape(ct_vec_last, (-1, self.args.emsize))   

        ct_vec = tf.reshape(outputs, (-1, self.args.emsize))
        outputs_shape = tf.shape(outputs)



        self.total_loss = 0.

        self.istarget = istarget = tf.reshape(tf.to_float(tf.not_equal(self.pos, 0)), [-1])

        _pos_emb = tf.nn.embedding_lookup(self.item_embedding, self.pos)
        pos_emb = tf.reshape(_pos_emb, (-1, self.args.emsize))
        _neg_emb = tf.nn.embedding_lookup(self.item_embedding, self.neg)
        neg_emb = tf.reshape(_neg_emb, (-1, self.args.neg_size, self.args.emsize))


        _pos_emb_dy = tf.nn.embedding_lookup(emb_list, self.pos_dy)
        pos_emb_dy = tf.reshape(_pos_emb_dy, (-1, self.args.emsize))
        _neg_emb_dy = tf.nn.embedding_lookup(emb_list, self.neg_dy)
        neg_emb_dy = tf.reshape(_neg_emb_dy, (-1, self.args.neg_size, self.args.emsize))


        pos_emb_joint = pos_emb + pos_emb_dy


        neg_emb_joint = neg_emb + neg_emb_dy
        
        
        temp_vec_neg = tf.tile(tf.expand_dims(ct_vec_last, [1]), [1, self.args.neg_size, 1]) 
        

        
        # assert self.args.neg_size == 1

        pos_logit = tf.reduce_sum(ct_vec_last * pos_emb_joint, -1)
        if self.args.neg_size == 1:
            a = tf.reduce_sum(temp_vec_neg * neg_emb_joint, -1) # both (?,1,100) #(?,100,100)
            neg_logit = tf.squeeze(a, 1)  # (?)
        else:
            a = tf.reduce_sum(temp_vec_neg * neg_emb_joint, -1)
            neg_logit = tf.reduce_mean(a, -1)
        loss = tf.reduce_sum(-tf.log(tf.sigmoid(pos_logit) + 1e-24) * istarget - \
                    tf.log(1 - tf.sigmoid(neg_logit) + 1e-24) * istarget \
                ) / tf.reduce_sum(istarget)

               
        ct_vec_batch = tf.tile(ct_vec_last, [self.test_neg_size+1, 1])
        self.test_item_batch = tf.placeholder(tf.int32, shape=(None, self.test_neg_size+1), name='test_item_batch')
        _test_item_emb_batch = tf.nn.embedding_lookup(self.item_embedding, self.test_item_batch)
        _test_item_emb_batch = tf.transpose(_test_item_emb_batch, perm=[1, 0, 2])
        test_item_emb_batch = tf.reshape(_test_item_emb_batch, (-1, self.args.emsize))

        self.test_item_batch_dy = tf.placeholder(tf.int32, shape=(None, self.test_neg_size+1), name='test_item_batch_dy')
        _test_item_emb_batch_dy = tf.nn.embedding_lookup(emb_list, self.test_item_batch_dy)
        _test_item_emb_batch_dy = tf.transpose(_test_item_emb_batch_dy, perm=[1, 0, 2])
        test_item_emb_batch_dy = tf.reshape(_test_item_emb_batch_dy, (-1, self.args.emsize))


        test_item_emb_batch_joint = test_item_emb_batch + test_item_emb_batch_dy

        self.test_logits_batch = tf.reduce_sum(ct_vec_batch*test_item_emb_batch_joint, -1) 
        self.test_logits_batch = tf.transpose(tf.reshape(self.test_logits_batch, [self.test_neg_size+1, tf.shape(self.inp)[0]]))

               
        self.loss = loss
        self.total_loss += loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss += sum(reg_losses)

        optimizer = tf.train.AdamOptimizer(self.lr)
        gvs = optimizer.compute_gradients(self.total_loss)
        capped_gvs = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -10., 10.), gv[1]], gvs)
        self.train_op = optimizer.apply_gradients(capped_gvs)        