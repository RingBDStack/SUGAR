import argparse
import configparser
import os
import re

import numpy as np
from keras.utils import to_categorical
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
######################################################################

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="MUTAG", dest='dataset')
args = parser.parse_known_args()[0]
######################################################################

os.chdir('./{}'.format(args.dataset))
print('dataset:', args.dataset)
config = configparser.ConfigParser()
config.read('.config')
getConfig = lambda x: config.get('config', x)
ds = getConfig('ds')
SUB_size = int(getConfig('SUB_size'))
min = int(getConfig('min'))

max_graph_nodes = 100
cate = int(getConfig('cate'))
class_size = int(getConfig('class_size'))


######################################################################

class graphParser(object):
    def __init__(self):
        self.ds = ds
        self.SUB_size = SUB_size
        self.min = min
        self.cate = cate
        self.class_size = class_size
        self.max_graph_nodes = max_graph_nodes

        #############################################################
        with open(self.ds + '_A.txt', 'r') as f:
            a = f.readlines()
            self.m = len(a)

        with open(self.ds + '_graph_labels.txt', 'r') as f:
            a = f.readlines()
            self.N = len(a)

        with open(self.ds + '_node_labels.txt', 'r') as f:
            a = f.readlines()
            self.nodeIndex2label = {}
            for index, item in enumerate(a):
                self.nodeIndex2label[index] = int(item)
            self.n = len(a)
        #############################################################

        print("m:{},N:{},n:{}".format(self.m, self.N, self.n))

    def ex_which_graph(self):
        d_n2g = {}
        with open(self.ds + '_graph_indicator.txt') as f:
            index = 0
            line = f.readline()
            while line:
                d_n2g[index] = int(line)
                index += 1
                line = f.readline()
        d_gns = {k: [] for k in range(self.N)}
        for it in d_n2g:
            d_gns[d_n2g[it] - 1].append(it)
        return d_gns

    def ex_edges(self):
        d_es = {k: [] for k in range(self.n)}
        with open(self.ds + '_A.txt') as f:
            line = f.readline()
            while line:
                line = line.replace(' ', '')
                s = re.search('[0-9]{1,}', line).group()
                d = re.search('[,][0-9]{1,}', line).group()
                d = d.strip(',')
                d_es[int(s) - 1].append(int(d) - 1)
                line = f.readline()
        adj = np.zeros((self.n, self.n))
        for it in d_es:
            for its in d_es[it]:
                adj[it][its] = 1
        return d_es, adj

    def ex_labels(self, d_gns):
        d_gl = {}
        glabs = np.zeros((self.N, self.class_size))
        with open(self.ds + '_graph_labels.txt') as f:
            #######################################################################
            old_cate_name2new_cate_new = {}
            new_cate_name = 0
            #######################################################################
            line = f.readline()
            index = 0
            while line:
                d_gl[index] = int(line)
                if d_gl[index] not in old_cate_name2new_cate_new.keys():
                    old_cate_name2new_cate_new[d_gl[index]] = new_cate_name
                    new_cate_name += 1
                glabs[index] = to_categorical(old_cate_name2new_cate_new[d_gl[index]], self.class_size)
                index += 1
                line = f.readline()
        return glabs

    def gen_adjmatrix(self, d_gns, adj_com):
        adjs = []
        index = 0
        for it in d_gns:
            gsize = len(d_gns[it])
            adj = adj_com[index:index + gsize, index:index + gsize]
            adjs.append(adj)
            index += gsize
        return adjs

    def gen_adj_onebyone(self, d_gns, adj_com):
        adjs = []
        index = 0
        for it in d_gns:
            gsize = len(d_gns[it])
            adj = []
            for nodeA in range(index, index + gsize):
                for nodeB in range(index + 1, index + gsize):
                    if adj_com[nodeA, nodeB] == 1:
                        adj.append(tuple(sorted([nodeA, nodeB])))
            index += gsize
            adjs.append(list(set(adj)))
        return adjs

    def nodeIndex2nodelabel(self, d_gns):
        nodeLabel = []

        for graph in d_gns.values():
            tempList = {x: self.nodeIndex2label[x] for x in graph}
            nodeLabel.append(tempList)
        return nodeLabel

    def main(self, save=True):
        d_gns = {}
        d_es = {}
        d_gl = {}
        glabs = []
        adj_com = []

        d_gns = self.ex_which_graph()
        d_es, adj_com = self.ex_edges()
        glabs = self.ex_labels(d_gns)
        nodeLabel = self.nodeIndex2nodelabel(d_gns)
        adjs = []
        adjs = self.gen_adjmatrix(d_gns, adj_com)
        adjs_onebyone = self.gen_adj_onebyone(d_gns, adj_com)

        self.d_gns = d_gns
        if save:
            f = open('edges.txt', 'w')
            f.write(str(d_es))
            f.close()

            f = open('nodes_in_graphs.txt', 'w')
            f.write(str(d_gns))
            f.close()

            np.save('graphs_label.npy', glabs)
            np.save('d_gns.npy', list(d_gns.values()))

            np.save('{}adjs.npy'.format(self.N), adjs)
            if not os.path.exists('graph'):
                os.mkdir('graph')
            np.save('graph/graphs_label.npy', glabs)
            np.save('graph/graph_node_labels.npy', nodeLabel)
            np.save('graph/adjs_onebyone.npy', adjs_onebyone)

        return adjs, d_es


class secondParse(graphParser):
    def __init__(self, adjs=None, dict_edges=None):
        super(secondParse, self).__init__()
        if not adjs:
            self.adjs = np.load('{}adjs.npy'.format(N), allow_pickle=True)
        else:
            self.adjs = adjs

        if not dict_edges:
            with open('edges.txt') as f:
                self.dict_edges = eval(f.read())
        else:
            self.dict_edges = dict_edges

    def bfs(self, adj, s):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        sub = []

        matrix = adj

        visited = [0 for _ in range(len(adj[0]))]

        queue = [s]

        visited[s] = 1

        node = queue.pop(0)

        sub.append(node)
        while True:
            for x in range(0, len(visited)):

                if matrix[node][x] == 1 and visited[x] == 0:
                    visited[x] = 1

                    queue.append(x)

            if len(queue) == 0:
                break
            else:

                newnode = queue.pop(0)
                node = newnode

                sub.append(node)
                if len(sub) == SUB_size:
                    sub.sort()
                    return sub
        sub.sort()
        return sub

    def gen_subs(self, adj):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        subs = []

        m = {}
        for s in range(len(adj[0])):
            if tuple(self.bfs(adj, s)) not in m.keys():
                subs.append(self.bfs(adj, s))
                m[tuple(self.bfs(adj, s))] = 1

        subs = subs[:min]
        return subs

    def gen_upperlevel_graph(self, sub):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        adj = np.zeros((len(sub), len(sub)))
        for i in range(len(sub)):
            for j in range(i + 1, len(sub)):
                same = [it for it in sub[i] if it in sub[j]]

                adj[i][j] = len(same)

            adj[i] = min_max_scaler.fit_transform(
                adj[i].reshape(-1, 1)).reshape(-1)

        for i in range(len(sub)):
            adj[i][i] = 1
        return adj

    def gen_subgraphs_adj_onebyone(self, all_adj, subs, nodeList):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        minNode = np.min(nodeList)
        adjs = []

        for key, sub in enumerate(subs):

            index = 0
            gsize = len(sub)
            adj = []
            for nodeA in sub:
                for nodeB in sub:
                    if all_adj[nodeA][nodeB] == 1 or all_adj[nodeB][nodeA] == 1:
                        adj.append(tuple(sorted([minNode + nodeA, minNode + nodeB])))
            adjs.append(list(set(adj)))
        return adjs

    def gen_subgraphs_adj(self, all_adj, subs):
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min

        adjs = []
        for sub in subs:
            adj = np.zeros((SUB_size, SUB_size))
            i = 0
            for i in range(len(sub)):
                for j in range(len(sub)):

                    if all_adj[sub[i]][sub[j]] == 1:
                        adj[i][j] = 1

                adj[i] = min_max_scaler.fit_transform(
                    adj[i].reshape(-1, 1)).reshape(-1)

            for i in range(len(sub)):
                adj[i][i] = 1
            adjs.append(adj.copy())
        return adjs

    def main(self, save=True):
        #######################################################################
        n = self.n
        m = self.m
        N = self.N
        SUB_size = self.SUB_size
        min = self.min
        adjs = self.adjs
        self.d_gns = np.load('d_gns.npy', allow_pickle=True)
        #######################################################################
        ADJs = []
        subs_of_graphs = []
        subadjs_of_graphs = []
        subadjs_of_graphs_onebyone = []
        min_max_scaler = preprocessing.MinMaxScaler()
        #######################################################################
        for i in range(len(adjs)):
            if i % 100 == 0:
                print(i)
            adj = adjs[i]
            sub_of_a_graph = self.gen_subs(adj)

            subs_of_graphs.append(sub_of_a_graph)
            subadjs_of_a_graph = self.gen_subgraphs_adj(
                adj, sub_of_a_graph)
            subadjs_of_a_graph_onebyone = self.gen_subgraphs_adj_onebyone(adj, sub_of_a_graph, self.d_gns[i])
            subadjs_of_graphs_onebyone.append(subadjs_of_a_graph_onebyone)
            subadjs_of_graphs.append(subadjs_of_a_graph)

            ADJ_of_a_graph = self.gen_upperlevel_graph(
                sub_of_a_graph)

            ADJs.append(ADJ_of_a_graph)

        if save:
            np.save('subs_of_graphs.npy', subs_of_graphs)
            np.save('subadjs_of_graphs.npy', subadjs_of_graphs)
            np.save('ADJs.npy', ADJs)
            if not os.path.exists('./graph'):
                os.mkdir('./graph')
            np.save('./graph/subadjs_of_graphs_onebyone.npy', subadjs_of_graphs_onebyone)

        sadjs = np.zeros((N, min, SUB_size, SUB_size))
        sus = np.zeros((N, min, SUB_size))
        As = np.zeros((N, min, min))

        if save:
            subs_of_graphs = np.load('subs_of_graphs.npy', allow_pickle=True)
            subadjs_of_graphs = np.load(
                'subadjs_of_graphs.npy', allow_pickle=True)
            ADJs = np.load('ADJs.npy', allow_pickle=True)

        for i in range(N):

            for j in range(len(ADJs[i])):
                As[i][j][:len(ADJs[i][j])][:len(
                    ADJs[i][j])] = ADJs[i][j].copy()
            for j in range(len(subadjs_of_graphs[i])):
                sadjs[i][j][:len(subadjs_of_graphs[i][j])][:len(
                    subadjs_of_graphs[i][j])] = subadjs_of_graphs[i][j].copy()
            for j in range(len(subs_of_graphs[i])):
                sus[i][j][:len(subs_of_graphs[i][j])
                ] = subs_of_graphs[i][j].copy()

        for i in range(N):
            for j in range(min):
                As[i][j][j] = 1
                for k in range(SUB_size):
                    sadjs[i][j][k][k] = 1

        if save:
            np.save('adj.npy', As)
            np.save('sub_adj.npy', sadjs)
            np.save('subindexs_of_graphs.npy', sus)
        return sus

    def third(self, sus=None):

        if sus == None:
            sus = np.load('subindexs_of_graphs.npy', allow_pickle=True)

        with open('nodes_in_graphs.txt') as f:
            dict_gn = eval(f.read())

        labs = self.ex_labels()
        features = np.zeros((self.N, self.min, self.SUB_size, self.cate))

        for i in range(features.shape[0]):
            for j in range(features.shape[1]):
                for k in range(features.shape[2]):
                    len_x = int(sus[i][j][k])

                    features[i][j][k] = labs[i][len_x]

        np.save('features.npy', features)

    def ex_labels(self):
        labs = {k: [] for k in range(self.N)}
        nodes = np.zeros(self.n)

        with open('nodes_in_graphs.txt') as f:
            dict_gn = eval(f.read())

        with open(self.ds + '_node_labels.txt') as f:
            line = f.readline()
            index = 0
            while line:
                nodes[index] = int(line)
                index += 1
                line = f.readline()

        i = 0
        for graph in dict_gn:
            labs[i] = to_categorical(
                [nodes[it] - 1 for it in dict_gn[graph]], self.cate)
            i += 1
        return labs


if __name__ == "__main__":
    paser = graphParser()

    adjs, d_es = paser.main(save=True)
    a = secondParse(adjs, d_es)

    sus = a.main()

    a.third()
    ######################

    # parser = graphParser()

    # d_gns = parser.ex_which_graph()
    # print(d_gns)
