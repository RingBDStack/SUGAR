import numpy as np
import tensorflow as tf

_LAYER_UIDS = {}


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    res = tf.layers.conv2d(x, y[1], [1, y[0]])
    return res[:, :, 0, :]


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


class Layer(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}


class GraphConvolution(Layer):
    """Graph convolution layer."""

    def __init__(self, input, adj_matrix, output_dim, dropout=0., act=tf.nn.relu, bias=False,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.dropout = dropout

        self.act = act
        self.adj_matrix = adj_matrix

        self.bias = bias
        self.input = input
        self.output_dim = output_dim

        with tf.variable_scope(self.name + '_vars'):
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

    def call(self):
        with tf.name_scope(self.name):
            outputs = self._call(self.input)
        return outputs

    def _call(self, inputs):
        # print(inputs)
        x = inputs
        x = tf.nn.dropout(x, self.dropout)

        # convolve
        tmp = tf.tile(tf.expand_dims(tf.reduce_sum(self.adj_matrix, axis=-1), -1), (1, 1, x.shape[-1]))
        x = tf.matmul(self.adj_matrix, x) / tmp
        # print(x.shape)
        self.weight = glorot([int(self.input.shape[-1]), int(self.output_dim)])
        pre_sup = tf.matmul(x, self.weight)

        output = pre_sup
        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

    def build(self):
        raise NotImplementedError


class GCN(Model):
    def __init__(self, x, adj_matrix, output_dim, dropout=0.5, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.input = x
        self.adj_matrix = adj_matrix
        self.dropout = dropout
        self.output_dim = output_dim

    def build(self):
        if len(self.output_dim) == 0:
            return self.input

        outputs = GraphConvolution(input=self.input,
                                   adj_matrix=self.adj_matrix,
                                   output_dim=self.output_dim[0],
                                   act=tf.nn.leaky_relu,
                                   dropout=self.dropout).call()

        for i in range(1, len(self.output_dim)):
            outputs = GraphConvolution(input=outputs,
                                       adj_matrix=self.adj_matrix,
                                       output_dim=self.output_dim[i],
                                       act=tf.nn.leaky_relu,
                                       dropout=self.dropout).call()

        return outputs
