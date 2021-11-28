import tensorflow as tf
import numpy as np
from HGCN.manifolds.hyperboloid import Hyperboloid
import HGCN.layers.hyp_layers as hyp_layers
import HGCN.manifolds as manifolds
flags = tf.app.flags
FLAGS = flags.FLAGS
import math

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):

    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Create a weight variable with Glorot & Bengio (AISTATS 2010)
    initialization.
    """
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)




class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs




# class HNN(Layer):
#     """
#     Hyperbolic Neural Networks.
#     """
#
#     def __init__(self, c, args):
#         super(HNN, self).__init__(c)
#         self.manifold = getattr(manifolds, args.manifold)()
#         dims, acts, _ = hyp_layers.get_dim_act_curv(args.act, args.num_layers, args.feat_dim, args.dim, None, args.device)
#         hnn_layers = []
#         for i in range(len(dims) - 1):
#             in_dim, out_dim = dims[i], dims[i + 1]
#             act = acts[i]
#             hnn_layers.append(
#                     hyp_layers.HNNLayer(
#                             self.manifold, in_dim, out_dim, self.c, args.dropout, act, args.bias)
#             )
#         self.layers = nn.Sequential(*hnn_layers)
#
#     def _call(self, x, adj):
#         x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
#         return super(HNN, self)(x_hyp, adj)



class HHGNN_conv(Layer):
    """
    Hyperbolic neural networks layer.
    """

    def __init__(self, manifold, in_features, out_features,adj, n_hyper, dropout=0., act=tf.nn.relu, use_bias=0, c=None):
        super(HHGNN_conv, self).__init__()
        self.manifold = getattr(manifolds, manifold)()
        self.linear = HypLinear(self.manifold, in_features, out_features, c, dropout, use_bias, n_hyper=2)
        self.hyp_act = HypAct(self.manifold, c, c, act)
        self.c = c
        self.adj = adj
        self.dropout = dropout
        self.n_hyper = n_hyper

    def _call(self, x):
        h = self.linear(x)
        mv = self.manifold.mobius_matvec(self.adj['G'], h, self.c)
        res = self.manifold.proj(mv, self.c)
        h_1 = self.hyp_act(res)
        h_2 = self.linear(h_1)
        mv = self.manifold.mobius_matvec(self.adj['G'], h_2, self.c)
        res = self.manifold.proj(mv, self.c)
        output_1 = self.hyp_act(res)

        x1 = tf.nn.dropout(output_1, 1 - self.dropout)
        y = self.manifold.mobius_matvec(self.adj['E'], x1, self.c)
        output_2 = self.hyp_act(y)


        return output_1, output_2

# class HHGNN_conv(Layer):
#     def __init__(self, args, input_dim, output_dim, adj, n_hyper, dropout=0. , act, **kwargs):
#         super(HHGNN_conv, self).__init__(**kwargs)
#         with tf.variable_scope(self.name + '_vars'):
#             k = 0
#             self.vars['weights_%d' %k] = weight_variable_glorot(input_dim, output_dim, name="weights_%d" %k)
#             for k in range(1, n_hyper+1):
#                 self.vars['weights_%d' %k] = weight_variable_glorot(output_dim, output_dim, name="weights_%d" %k)
#         self.dropout = dropout
#         self.adj = adj
#         self.act = HypAct(manifolds)
#         self.n_hyper = n_hyper
#         self.manifold = getattr(manifolds, 'PoincareBall')()
#
#
#
#     def _call(self, inputs):
#         x = inputs
#         y = tf.sparse_tensor_dense_matmul(self.adj['E'], x)
#         x_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c), c=self.c)
#         y_hyp = self.manifold.proj(self.manifold.expmap0(self.manifold.proj_tan0(y, self.c), c=self.c), c=self.c)
#
#         # first layer - no dropout
#         k = 0
#         x_hyp = self.manifold.mobius_matvec(self.vars['weights_%d' % k], x_hyp, self.c)
#         mv_x = self.manifold.mobius_matvec(x_hyp, self.adj['G'])
#         res = self.manifold.proj(mv_x, self.c)
#
#
#         mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
#
#
#
#         res = self.manifold.proj(mv, self.c)
#
#         #hyperbolic linear
#         drop_weight = F.dropout(self.weight, self.dropout, training=self.training)
#         mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
#         res = self.manifold.proj(mv, self.c)
#         # hyperbolic linear
#
#         return res


class HGNN_conv(Layer):
    """Basic hypergraph convolution layer."""
    def __init__(self, input_dim, output_dim, adj, n_hyper, dropout=0., act=tf.nn.relu, **kwargs):
        super(HGNN_conv, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            k = 0
            self.vars['weights_%d' %k] = weight_variable_glorot(input_dim, output_dim, name="weights_%d" %k)
            for k in range(1, n_hyper+1):
                self.vars['weights_%d' %k] = weight_variable_glorot(output_dim, output_dim, name="weights_%d" %k)
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.n_hyper = n_hyper

    def _call(self, inputs):
        x = inputs
        # first layer - no dropout
        k = 0
        x = tf.matmul(x, self.vars['weights_%d' %k])
        y = tf.sparse_tensor_dense_matmul(self.adj['E'], x)
        x = tf.sparse_tensor_dense_matmul(self.adj['G'], x)

        x = self.act(x)

        for k in range(1, self.n_hyper):
            x = tf.nn.dropout(x, 1-self.dropout)
            x = tf.matmul(x, self.vars['weights_%d' %k])
            x = tf.sparse_tensor_dense_matmul(self.adj['G'], x)
            x = self.act(x)
        k = self.n_hyper
        x1 = tf.nn.dropout(x, 1-self.dropout)
        y = tf.sparse_tensor_dense_matmul(self.adj['E'], x1)
        y = self.act(y)
        return x,y

class HypAct(Layer):
    """
    Hyperbolic activation layer.
    """

    def __init__(self, manifold, c_in, c_out, act):
        super(HypAct, self).__init__()
        self.manifold = manifold
        self.c_in = c_in
        self.c_out = c_out
        self.act = act

    def _call(self, x):
        xt = self.act(self.manifold.logmap0(x, c=self.c_in))
        xt = self.manifold.proj_tan0(xt, c=self.c_out)
        return self.manifold.proj(self.manifold.expmap0(xt, c=self.c_out), c=self.c_out)

    def extra_repr(self):
        return 'c_in={}, c_out={}'.format(
            self.c_in, self.c_out
        )

class HypLinear(Layer):
    """
    Hyperbolic linear layer. # can change dimension
    """

    def __init__(self, manifold, in_features, out_features, c, dropout, use_bias, n_hyper):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        self.c = c
        self.dropout = dropout
        self.use_bias = use_bias
        self.bias = tf.Tensor(out_features)
        self.weight = tf.Tensor(out_features, in_features)
        with tf.variable_scope(self.name + '_vars'):
            k = 0
            self.vars['weights_%d' %k] = weight_variable_glorot(in_features, out_features, name="weights_%d" %k)
            for k in range(1, n_hyper+1):
                self.vars['weights_%d' %k] = weight_variable_glorot(in_features, out_features, name="weights_%d" %k)
        # self.reset_parameters()

    # def reset_parameters(self):
        # init.xavier_uniform_(self.weight, gain=math.sqrt(2))
        # init.constant_(self.bias, 0)

    def _call(self, x):
        drop_weight = tf.nn.dropout(x, 1 - self.dropout)
        mv = self.manifold.mobius_matvec(drop_weight, x, self.c)
        res = self.manifold.proj(mv, self.c)
        if self.use_bias:
            bias = self.manifold.proj_tan0(self.bias.view(1, -1), self.c)
            hyp_bias = self.manifold.expmap0(bias, self.c)
            hyp_bias = self.manifold.proj(hyp_bias, self.c)
            res = self.manifold.mobius_add(res, hyp_bias, c=self.c)
            res = self.manifold.proj(res, self.c)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, c={}'.format(
            self.in_features, self.out_features, self.c
        )