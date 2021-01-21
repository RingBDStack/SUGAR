# coding: utf-8

import tensorflow as tf

conv1d = tf.layers.conv1d


def fcn_layer(
        inputs,
        input_dim, 
        output_dim,
        activation=None): 

    W = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))
    XWb = tf.matmul(inputs, W) + b  # Y=WX+B

    if (activation == None):
        outputs = XWb
    else:
        outputs = activation(XWb)
    return outputs


def relu(x, alpha=0., max_value=None):
    '''
    ReLU.
    alpha: slope of negative section.
    '''
    negative_part = tf.nn.relu(-x)
    x = tf.nn.leaky_relu(x)
    if max_value is not None:
        x = tf.clip_by_value(x, tf.cast(0., dtype=tf.float32),
                             tf.cast(max_value, dtype=tf.float32))
    x -= tf.constant(alpha, dtype=tf.float32) * negative_part
    return x


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.5, residual=False):
    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        # logits = tf.matmul(f_1 , tf.transpose(f_2, [0, 2, 1]))
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits))  # )
        # coefs = tf.matmul(tf.matrix_diag(1/(tf.reduce_sum(logits,axis = -1)+0.01)),
        # logits)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        # ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(vals)  # activation


def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(
                                        coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                seq_fts = ret + seq

        return activation(ret)  # activation


class BaseGAttN:
    def loss(logits, labels, nb_classes, class_weights):
        sample_wts = tf.reduce_sum(tf.multiply(
            tf.one_hot(labels, nb_classes), class_weights), axis=-1)
        xentropy = tf.multiply(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits), sample_wts)
        return tf.reduce_mean(xentropy, name='xentropy_mean')

    def training(loss, lr, l2_coef):
        # weight decay
        vars = tf.trainable_variables()
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars if v.name not
                           in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef
        opt = tf.train.AdamOptimizer(learning_rate=lr)

        train_op = opt.minimize(loss + lossL2)

        return train_op

    def preshape(logits, labels, nb_classes):
        new_sh_lab = [-1]
        new_sh_log = [-1, nb_classes]
        log_resh = tf.reshape(logits, new_sh_log)
        lab_resh = tf.reshape(labels, new_sh_lab)
        return log_resh, lab_resh

    def confmat(logits, labels):
        preds = tf.argmax(logits, axis=1)
        return tf.confusion_matrix(labels, preds)


class GAT(BaseGAttN):
    def inference(self, inputs, nb_classes, bias_mat, hid_units,
                  n_heads, activation=tf.nn.elu, residual=False, k=0.5):

        select_num = tf.cast(inputs.shape[1].value * k, dtype=tf.int32)

        # mean_sum = tf.reduce_sum(tf.square(inputs), -1)
        p = tf.Variable(tf.truncated_normal([int(inputs.shape[-1]), 1], stddev=0.1))
        mean_sum = tf.reshape(tf.matmul(inputs, p) / tf.reduce_sum(tf.square(p)), [-1, int(inputs.shape[1])])

        a_top, a_top_idx = tf.nn.top_k(mean_sum, select_num)

        a_top_1, a_top_idx_1 = tf.nn.top_k(mean_sum, inputs.shape[1])

        a_shape = tf.shape(mean_sum)
        a_top_sm = a_top * 0 + 1

        a_row_idx = tf.tile(tf.range(a_shape[0])[:, tf.newaxis], (1, select_num))
        """
        a_row_idx = [array([[ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
       [ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1],
       ...
       [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]],
      dtype=int32)]
        """
        scatter_idx = tf.stack([a_row_idx, a_top_idx], axis=-1)
        result = tf.scatter_nd(scatter_idx, a_top_sm, a_shape)
        a_index = tf.tile(tf.expand_dims(result, -1), (1, 1, inputs.shape[-1]))
        c_index = a_index
        inputs = a_index * inputs

        attns = []
        for _ in range(n_heads[0]):
            attns.append(attn_head(inputs, bias_mat=bias_mat,
                                   out_sz=hid_units[0], activation=activation, residual=False))
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            attns = []
            for _ in range(n_heads[i]):
                attns.append(attn_head(h_1, bias_mat=bias_mat,
                                       out_sz=hid_units[i], activation=activation, residual=residual))
            h_1 = tf.concat(attns, axis=-1)

        a_index = tf.tile(tf.expand_dims(result, -1), (1, 1, h_1.shape[-1]))
        h_1 = a_index * h_1

        logits = tf.layers.dense(
            inputs=h_1, units=nb_classes, activation=tf.nn.leaky_relu)
        a_index = tf.tile(tf.expand_dims(result, -1), (1, 1, logits.shape[-1]))
        logits = a_index * logits
        return a_index, h_1, logits, inputs, select_num, a_top_idx_1
