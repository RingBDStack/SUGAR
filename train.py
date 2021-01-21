import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import random
import tensorflow as tf
import numpy as np
import json
from GCN import *
import SAGE
import tqdm
import layers
import time
from QLearning import *
from env import GNN_env
import warnings


# num_info
num_infoGraph = 1

hid_units = [16]
n_heads = [6, 1]
residual = False
nonlinearity = tf.nn.leaky_relu
attention_model = layers.GAT()


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def data_preprocess(dataset):
    adj = np.load('dataset/' + dataset + '/adj.npy', allow_pickle=True)
    feature = np.load('dataset/' + dataset +
                      '/features.npy', allow_pickle=True)
    subadj = np.load('dataset/' + dataset + '/sub_adj.npy', allow_pickle=True)
    label = np.load('dataset/' + dataset +
                    '/graphs_label.npy', allow_pickle=True)

    new_label = np.array([np.argmax(one_hot) for one_hot in label])
    label_idx = []
    for i in range(label.shape[-1]):
        tmp = np.where(new_label == i)
        label_idx.append(tmp)
    return feature, new_label, label.shape[1], np.ones([adj.shape[0], adj.shape[1]]), subadj, adj, adj.shape[1], \
           subadj.shape[-1], label_idx


def get_dsi_idx(label_idx, train_t):
    dsi_idx = []
    for i in range(num_infoGraph):
        for t in train_t:
            temp = list(range(len(label_idx)))
            temp.remove(t)
            temp_label = label_idx[(random.sample(temp, 1)[0])]
            dsi_idx.append(random.sample(list(temp_label[0]), 1)[0])
    return dsi_idx


def divide_train_test(data, label, sub_adj, sub_mask, test_begin_idx, test_end_idx, label_idx):
    data_size = data.shape[0]
    train_x = np.concatenate([data[0:test_begin_idx], data[test_end_idx:data_size]])
    train_t = np.concatenate([label[0:test_begin_idx], label[test_end_idx:data_size]])
    train_sadj = np.concatenate([sub_adj[0:test_begin_idx], sub_adj[test_end_idx:data_size]])
    train_mask = np.concatenate([sub_mask[0:test_begin_idx], sub_mask[test_end_idx:data_size]])

    dsi_idx = get_dsi_idx(label_idx, train_t)
    train_x_dsi = data[dsi_idx]
    train_t_dsi = label[dsi_idx]
    train_sadj_dsi = sub_adj[dsi_idx]
    train_mask_dsi = sub_mask[dsi_idx]

    test_x = data[test_begin_idx:test_end_idx]
    test_t = label[test_begin_idx:test_end_idx]
    test_sadj = sub_adj[test_begin_idx:test_end_idx]
    test_mask = sub_mask[test_begin_idx:test_end_idx]

    dsi_idx = get_dsi_idx(label_idx, test_t)
    test_x_dsi = data[dsi_idx]
    test_t_dsi = label[dsi_idx]
    test_sadj_dsi = sub_adj[dsi_idx]
    test_mask_dsi = sub_mask[dsi_idx]

    return train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi, \
           test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi


def load_batch(x, sadj, t, mask, train_x_dsi, train_sadj_dsi, train_t_dsi, train_mask_dsi, i, batch_size):
    data_size = x.shape[0]
    if i + batch_size > data_size:
        index = [j for j in range(i, data_size)]
        dsi_index = [j for j in range(i * num_infoGraph, data_size * num_infoGraph)]
    else:
        index = [j for j in range(i, i + batch_size)]
        dsi_index = [j for j in range(i * num_infoGraph, (i + batch_size) * num_infoGraph)]
    return x[index], sadj[index], t[index], mask[index], train_x_dsi[dsi_index], train_sadj_dsi[dsi_index], train_t_dsi[
        dsi_index], train_mask_dsi[dsi_index]


class HGANP(object):
    def __init__(self, session, embedding, ncluster, num_subg, subg_size, batch_size, learning_rate, momentum):
        self.sess = session
        self.ncluster = ncluster
        self.embedding = embedding
        self.num_subg = num_subg
        self.subg_size = subg_size
        self.batch_size = batch_size
        self.output_dim = [32]
        self.GIN_dim = [16]
        self.SAGE_dim = [32]
        self.sage_k = 1
        self.lr = learning_rate
        self.mom = momentum

        self.build_placeholders()
        self.forward_propagation()
        train_var = tf.compat.v1.trainable_variables()
        self.l2 = tf.contrib.layers.apply_regularization(
            tf.contrib.layers.l2_regularizer(0.01), train_var)
        self.pred = tf.to_int32(tf.argmax(self.probabilities, 1))
        correct_prediction = tf.equal(self.pred, self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        self.optimizer = tf.compat.v1.train.MomentumOptimizer(
            self.lr, self.mom).minimize(self.loss + self.l2)
        self.init = tf.global_variables_initializer()
        self.saver = tf.compat.v1.train.Saver(tf.global_variables())


    def mlp(self,
            inputs,
            input_dim,
            output_dim,
            activation=None):

        W = tf.Variable(tf.truncated_normal(
            [input_dim, output_dim], stddev=0.1))
        b = tf.Variable(tf.zeros([output_dim]))
        XWb = tf.matmul(inputs, W) + b

        if (activation == None):
            outputs = XWb
        else:
            outputs = activation(XWb)
        return outputs

    def build_placeholders(self):
        self.sub_adj = (tf.compat.v1.placeholder(tf.float32, shape=(
            None, self.num_subg, self.subg_size, self.subg_size)))
        self.sub_feature = (tf.compat.v1.placeholder(tf.float32, shape=(
            None, self.num_subg, self.subg_size, self.embedding)))
        self.sub_feature_dsi = (tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.num_subg, self.subg_size, self.embedding)))
        self.sub_mask = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.num_subg))
        self.sub_mask_dsi = tf.compat.v1.placeholder(
            tf.float32, shape=(None, self.num_subg))
        self.labels = tf.compat.v1.placeholder(tf.int32, shape=(None))
        self.label_mi = tf.compat.v1.placeholder(
            tf.int32, shape=(None, (num_infoGraph + 1) * self.num_subg))
        self.lr = tf.compat.v1.placeholder(tf.float32, [], 'learning_rate')
        self.mom = tf.compat.v1.placeholder(tf.float32, [], 'momentum')
        self.dropout = tf.compat.v1.placeholder_with_default(0.5, shape=())
        self.top_k = tf.compat.v1.placeholder_with_default(0.5, shape=())

    def sub_GCN(self):
        gcn_outs = []
        W = tf.Variable(tf.random.truncated_normal(
            [self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = GCN(
                self.sub_feature[:, i, :, :], self.d_matrix, self.output_dim, dropout=0.5).build()
            gcn_out = tf.matmul(tf.transpose(gcn_out, [0, 2, 1]), W)
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        W_dsi = tf.Variable(tf.random.truncated_normal(
            [self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = GCN(
                self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.output_dim, dropout=0.5).build()
            gcn_out_dsi = tf.matmul(tf.transpose(
                gcn_out_dsi, [0, 2, 1]), W_dsi)
            gcn_outs_dsi.append(tf.reshape(
                gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_GAT(self):
        gcn_outs = []
        W = tf.Variable(tf.random.truncated_normal(
            [self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            _, gcn_out, _, _, _, _ = attention_model.inference(self.sub_feature[:, i, :, :], self.ncluster, 0,
                                                               self.output_dim, n_heads, nonlinearity,
                                                               residual, 1)
            gcn_out = tf.matmul(tf.transpose(gcn_out, [0, 2, 1]), W)
            gcn_outs.append(tf.reshape(gcn_out, [-1, 1, gcn_out.shape[1]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        W_dsi = tf.Variable(tf.random.truncated_normal(
            [self.subg_size, 1], stddev=0.1))
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            _, gcn_out_dsi, _, _, _, _ = attention_model.inference(self.sub_feature_dsi[:, i, :, :], self.ncluster, 0,
                                                                   self.output_dim, n_heads, nonlinearity,
                                                                   residual, 1)
            gcn_out_dsi = tf.matmul(tf.transpose(
                gcn_out_dsi, [0, 2, 1]), W_dsi)
            gcn_outs_dsi.append(tf.reshape(
                gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_GIN(self):
        gcn_outs = []
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = GCN(
                self.sub_feature[:, i, :, :], self.d_matrix, self.output_dim, dropout=0.5).build()
            for i in range(1):
                gcn_out = self.mlp(
                    gcn_out, self.output_dim[i], self.GIN_dim[i])
            gcn_outs.append(tf.reshape(
                gcn_out, [-1, 1, gcn_out.shape[1] * gcn_out.shape[2]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = GCN(
                self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.output_dim, dropout=0.5).build()
            for i in range(1):
                gcn_out_dsi = self.mlp(
                    gcn_out_dsi, self.output_dim[i], self.GIN_dim[i])
            gcn_outs_dsi.append(tf.reshape(
                gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1] * gcn_out_dsi.shape[2]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def sub_SAGE(self):
        gcn_outs = []
        for i in range(self.num_subg):
            self.d_matrix = self.sub_adj[:, i, :, :]
            gcn_out = SAGE.GCN(
                self.sub_feature[:, i, :, :], self.d_matrix, self.SAGE_dim, dropout=0.5).build()
            gcn_outs.append(tf.reshape(
                gcn_out, [-1, 1, gcn_out.shape[1] * gcn_out.shape[2]]))
        self.gcn_result = tf.concat(gcn_outs, 1)

        gcn_outs_dsi = []
        for i in range(self.num_subg):
            self.d_matrix_dsi = self.sub_adj[:, i, :, :]
            gcn_out_dsi = SAGE.GCN(
                self.sub_feature_dsi[:, i, :, :], self.d_matrix_dsi, self.SAGE_dim, dropout=0.5).build()
            gcn_outs_dsi.append(tf.reshape(
                gcn_out_dsi, [-1, 1, gcn_out_dsi.shape[1] * gcn_out_dsi.shape[2]]))
        self.gcn_result_dsi = tf.concat(gcn_outs_dsi, 1)

    def graph_gat(self):
        self.embedding_origin = self.gcn_result
        self.index, self.gatembedding, self.gat_result, self.embedding_topk, self.select_num, self.a_index = attention_model.inference(
            self.gcn_result, self.ncluster, 0,
            hid_units, n_heads, nonlinearity,
            residual, self.top_k)

        self.index_dsi, self.gatembedding_dsi, self.gat_result_dsi, _, __, _ = attention_model.inference(
            self.gcn_result_dsi, self.ncluster,
            0, hid_units, n_heads, nonlinearity,
            residual, self.top_k)

    def bilinear(self, x, y, out_dim, flag):
        w = tf.ones([out_dim, x.shape[-1], y.shape[-1]])
        w = tf.expand_dims(w, 0)
        w = tf.tile(w, tf.stack([x.shape[1], 1, 1, 1]))

        x = tf.expand_dims(x, 2)
        x = tf.expand_dims(x, 4)
        x = tf.tile(x, tf.stack([1, 1, out_dim, 1, y.shape[-1]]))
        tmp = tf.reduce_sum(tf.multiply(x, w), 3)

        y = tf.expand_dims(y, 2)
        y = tf.tile(y, tf.stack([1, 1, out_dim, 1]))
        if flag:
            tmp = tf.tile(tmp, (num_infoGraph, 1, 1, 1))
        out = tf.reduce_sum(tf.multiply(tmp, y), 3)

        return out

    def forward_propagation(self):
        with tf.variable_scope('sub_gcn'):
            if sg_encoder == 'GCN':
                self.sub_GCN()
            elif sg_encoder == 'GAT':
                self.sub_GAT()
            elif sg_encoder == 'GIN':
                self.sub_GIN()
            elif sg_encoder == 'SAGE':
                self.sub_SAGE()

        with tf.variable_scope('graph_gat'):
            self.graph_gat()

        with tf.variable_scope('fn'):
            vote_layer = tf.reduce_sum(self.gat_result, axis=1)
            self.loss = tf.compat.v1.losses.sparse_softmax_cross_entropy(
                labels=self.labels, logits=vote_layer)

            global_embedding = tf.reduce_sum(self.gatembedding, axis=1)
            global_embedding = tf.tile(tf.expand_dims(
                global_embedding, 1), (1, self.num_subg, 1))
            sc = self.bilinear(global_embedding, self.gatembedding, 1, False)
            sc_dsi = self.bilinear(global_embedding, self.gatembedding_dsi, 1, True)
            sc_dsi = tf.reshape(sc_dsi, [-1, sc.shape[1] * num_infoGraph, 1])
            self.sc = tf.sigmoid(tf.concat([sc, sc_dsi], 1))
            self.sc = tf.concat([self.sc, 1 - self.sc], 2)
            self.loss += MI_loss * \
                         tf.losses.sparse_softmax_cross_entropy(
                             labels=self.label_mi, logits=self.sc)
            self.probabilities = tf.nn.softmax(
                vote_layer, name="probabilities")
            self.sub_true = tf.to_int32(tf.argmax(self.gat_result, 2))
            self.tmp_labels = tf.tile(tf.expand_dims(
                self.labels, 1), (1, self.num_subg))
            self.RL_reward = tf.reduce_mean(
                tf.cast(tf.equal(self.sub_true, self.tmp_labels), "float"))

    def train(self, batch_x, batch_x_dsi, batch_adj, batch_t, batch_t_mi, batch_mask, learning_rate=1e-3, momentum=0.9,
              k=0.5):
        feed_dict = {
            self.sub_feature: batch_x,
            self.sub_feature_dsi: batch_x_dsi,
            self.sub_adj: batch_adj,
            self.labels: batch_t,
            self.label_mi: batch_t_mi,
            self.sub_mask: batch_mask,
            self.sub_mask_dsi: batch_mask,
            self.lr: learning_rate,
            self.mom: momentum,
            self.top_k: k
        }
        _, loss, acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index = self.sess.run(
            [self.optimizer, self.loss,
             self.accuracy, self.pred,
             self.sub_true, self.gcn_result,
             self.gat_result, self.sub_feature,
             self.index, self.embedding_origin,
             self.embedding_topk, self.select_num, self.a_index], feed_dict=feed_dict)
        return loss, acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index

    def evaluate(self, batch_x, batch_x_dsi, batch_adj, batch_t, batch_t_mi, batch_mask, k):
        feed_dict = {
            self.sub_feature: batch_x,
            self.sub_feature_dsi: batch_x_dsi,
            self.sub_adj: batch_adj,
            self.labels: batch_t,
            self.label_mi: batch_t_mi,
            self.sub_mask: batch_mask,
            self.sub_mask_dsi: batch_mask,
            self.top_k: k
        }
        acc, pred, index, rl_reward, embedding_origin, embedding_topk, select_num, a_index = self.sess.run(
            [self.accuracy, self.pred, self.index, self.RL_reward, self.embedding_origin, self.embedding_topk,
             self.select_num, self.a_index], feed_dict=feed_dict)
        return acc, pred, index, rl_reward, embedding_origin, embedding_topk, select_num, a_index


def main(params):
    ###############################################
    global max_pool
    global MI_loss
    global sg_encoder
    folds = params['folds']
    dataset = params['dataset']
    num_epochs = params['num_epochs']
    max_pool = params['max_pool']
    batch_size = params['batch_size']
    learning_rate = params['learning_rate']
    momentum = params['momentum']
    MI_loss = params['MI_loss']
    sg_encoder = params['sg_encoder']
    k = params['start_k']
    ###############################################
    feature, label, ncluster, sub_mask, sub_adj, vir_adj, num_subg, subg_size, label_idx = data_preprocess(dataset)
    test_size = int(feature.shape[0] / folds)
    train_size = feature.shape[0] - test_size
    learning_rate = learning_rate
    with tf.Session() as sess:
        net = HGANP(sess, feature.shape[-1], ncluster, num_subg,
                    subg_size, batch_size, learning_rate, momentum)
        accs = []
        for fold in range(folds):
            sess.run(tf.global_variables_initializer())
            vir_acc_fold = []
            if fold < folds - 1:
                train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi, \
                test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi \
                    = divide_train_test(feature, label, sub_adj, sub_mask,
                                        fold * test_size,
                                        fold * test_size + test_size, label_idx)
            else:
                train_x, train_t, train_sadj, train_mask, train_x_dsi, train_t_dsi, train_sadj_dsi, train_mask_dsi, \
                test_x, test_t, test_sadj, test_mask, test_x_dsi, test_t_dsi, test_sadj_dsi, test_mask_dsi \
                    , = divide_train_test(feature, label, sub_adj, sub_mask,
                                          feature.shape[0] - test_size,
                                          feature.shape[0], label_idx)
            max_fold_acc = 0
            k_step_value = round(0.5 / net.num_subg, 4)
            env = GNN_env(action_value=k_step_value,
                           subgraph_num=net.num_subg, initial_k=k)
            RL = QLearningTable(actions=list(range(env.n_actions)), learning_rate=0.02)
            k_record = []
            eva_acc_record = []

            tbar = tqdm.tqdm(range(num_epochs))
            train_acc_record = []
            train_loss_record = []
            endingRLEpoch = 0

            for epoch in tbar:
                train_loss = 0
                train_acc = 0
                batch_num = 0
                idx = np.random.permutation(feature.shape[2])
                for i in range(0, train_size, batch_size):
                    x_batch, sadj_batch, t_batch, mask_batch, x_batch_dsi, sadj_batch_dsi, t_batch_dsi, mask_batch_dsi \
                        = load_batch(train_x, train_sadj, train_t, train_mask, \
                                     train_x_dsi, train_sadj_dsi, train_t_dsi, train_mask_dsi, i, batch_size)
                    t_batch_mi = [[1] * num_subg + [0] * num_subg * num_infoGraph] * len(t_batch)

                    loss, acc, pred, sub_true, gcn_result, gat_result, sub_feature, index, embedding_origin, embedding_topk, select_num, a_index = net.train(
                        x_batch, x_batch_dsi,
                        sadj_batch, t_batch,
                        t_batch_mi, mask_batch,
                        learning_rate, momentum,
                        k)
                    limited_epoch = 20
                    delta_k = 0.04
                    if epoch >= 100 and (not isTerminal(k_record, limited_epochs=limited_epoch, delta_k=delta_k)):
                        k, reward = run_QL(env, RL, net, x_batch, x_batch_dsi, sadj_batch, t_batch, t_batch_mi, mask_batch, acc)
                        k_record.append(round(k, 4))
                        endingRLEpoch = epoch
                    else:
                        k_record.append(round(k, 4))

                    batch_num += 1
                    train_loss += loss
                    train_acc += acc

                    batch_num += 1
                    if i == 0:
                        all_mask = sub_true
                    else:
                        all_mask = np.concatenate([all_mask, sub_true], 0)
                test_t_mi = [[1] * num_subg + [0] * num_subg * num_infoGraph] * len(test_t)
                test_dsi = test_x[:, :, idx, :]
                eva_acc, eva_pred, eva_index, _, eva_embedding_origin, eva_embedding_topk, eva_selecnum, eva_a_index = net.evaluate(
                    test_x, test_dsi, test_sadj, test_t, test_t_mi, test_mask, k)

                if eva_acc > max_fold_acc:
                    max_fold_acc = eva_acc
                    vir_acc_fold.append(eva_acc)

                # np.save(f'{limited_epoch}_{delta_k}_{fold}.npy', k_record)

                train_loss_record.append(train_loss / batch_num)
                train_acc_record.append(eva_acc)
                tbar.set_description_str("folds {}/{}".format(fold + 1, folds))
                tbar.set_postfix_str("k:{:.2f}, loss: {:.2f}, best_acc:{:.2f}, RL:{}".format(k, train_loss / batch_num, max_fold_acc, endingRLEpoch))

                try:
                    eva_acc_record.append(eva_acc)
                except:
                    eva_acc_record = [eva_acc]

            accs.append(max_fold_acc)
        accs = np.array(accs)
        mean = np.mean(accs) * 100
        std = np.std(accs) * 100
        ans = {
            "mean": mean,
            "std": std
        }
        ####################
        return ans


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="MUTAG")
    # parser.add_argument('--num_info', type=float, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--max_pool', type=float, default=0.06)
    parser.add_argument('--momentum', type=float, default=0.8)
    parser.add_argument('--num_epoch', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--sg_encoder', type=str, default='GCN')
    parser.add_argument('--MI_loss', type=float, default=0.8)
    parser.add_argument('--start_k', type=float, default=0.8)

    args = parser.parse_known_args()[0]

    params = {
    'dataset' : args.dataset,
    'folds' : 3,
    'num_epochs' : args.num_epoch,
    'batch_size' : args.batch_size,
    'max_pool' : args.max_pool,
    'learning_rate' : args.lr,
    'momentum' : args.momentum,
    'sg_encoder' : args.sg_encoder,
    'MI_loss' : args.MI_loss,
    'start_k' : args.start_k,
    }

    ans = main(params)
