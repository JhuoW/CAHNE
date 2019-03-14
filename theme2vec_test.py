import tensorflow as tf
import numpy as np
import os
import pickle
from theme2vecDataSet import theme2vecDataSet
import random
import theme

# dataset = "hepth"
# graph_path = '../../data/hepth/graph.txt'
# text_path = '../../data/hepth/data.txt'
# cache_filename = "../cache/" + dataset + ".pkl"

dataset = 'hepth'
graph_path = 'graph_train.txt'
text_path = '../../data/hepth/data.txt'
cache_filename = 'cache/' + dataset + ".pkl"


class theme2vec:
    def __init__(self, neg_size=10, batch_size=39, epoch_num=100, walk_length=2, random_walk_epochs=5, embed_size=50,
                 sentence_len=100):
        self.dataset = theme2vecDataSet(text_path, graph_path, batch_size, sentence_len)
        nodes, node_size, graph = self.dataset.read_edges()
        self.nodes = nodes  # 已排序
        self.node_size = node_size
        self.neg_size = neg_size
        self.sentence_len = sentence_len
        self.batch_size = batch_size
        self.epochs = epoch_num
        self.embed_size = embed_size
        self.walk_length = walk_length
        root_nodes = [i for i in nodes]
        trees = None
        if os.path.isfile(cache_filename):
            print("reading BFS-trees from cache...")
            pickle_file = open(cache_filename, 'rb')
            trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(cache_filename, 'wb')
            trees = self.dataset.buildTree(root_nodes, graph)
            pickle.dump(trees, pickle_file)
            pickle_file.close()
        self.trees = trees
        self.text, self.num_vocab = self.dataset.text, self.dataset.num_vocab
        self.all_node = len(self.text)
        # self.text.shape:[1038,300]
        self.textdict = dict()
        for i in range(self.node_size):
            self.textdict[i] = self.text[i]

        self.neighborhood = graph  # {440:[50,103,...], 50:[440,222,257,...],...}
        # random walk:  从每个更节点向下随机游走walk_length层 每个节点做random_walk_epochs次这样的操作
        sample = dict()
        """
        for i in nodes:
            sample[i] = list()
            nei = graph[i]
            node = random.sample(nei, 1)
            for j in range(walk_length):
                k = node[0]
                sample[i].append(node[0])
                nei = graph[node[0]]
                if len(nei) != 1:
                    node = random.sample(nei, 1)
                    while node[0] == i or node[0] in sample[i]:
                        node = random.sample(nei, 1)
                else:
                    break
        print(sample)
        """
        # print(self.trees[207]) {207: [207, 26], 26: [207]}
        sample_node = dict()
        # {0: [279, 9], 1: [155, 112], 2: [73], 3: [527, 628], 4: [260, 106], 5: [59, 346, 173],...}
        # 采样节点
        for i in nodes:
            sample = self.sampleFromTree(i)
            sample_node[i] = sample
        # len:1026
        node_theme = dict()
        # {0: [398, 151, 353], 1: [155, 112, 1], 2: [73, 2, 2], 3: [527, 628, 3], 4: [260, 106, 4], ...}
        for i in nodes:
            whole_text = sample_node[i]
            while len(whole_text) < walk_length:
                whole_text.append(i)
            node_theme[i] = whole_text

        self.theme = dict()
        for i in nodes:
            thee = node_theme[i]
            theme_i = []
            for j in thee:
                text = self.text[j]
                theme_i.append(list(text))
            self.theme[i] = theme_i
        print(len(self.theme))
        # self.theme = np.array(self.theme)  # shape[1380,2,100]

    def sampleFromTree(self, root):
        # print(trees)    {root_id:{tree of root_id},root_id2:{tree of root_id2},...}
        # tree of root_id:  {0:[0,1,2,3,4,5], 1:[0,8,22,56],2:[0,...],...}  tree
        current_node = root
        previous_node = -1
        sample = []
        is_root = True
        tree = self.trees[root]  # 根节点的邻居
        is_root = True
        while len(sample) < self.walk_length:
            if is_root:
                if len(tree[root]) == 1:
                    sample.append(root)
                    break
                node = random.sample(tree[root][1:], 1)[0]  # 若root 27 则 node为26
                sample.append(node)
                is_root = False
                root = node
            else:
                if len(tree[root]) == 1:
                    break
                node = random.sample(tree[root][1:], 1)[0]
                sample.append(node)
                root = node

        return sample

    def train(self):
        with tf.Graph().as_default():
            sess = tf.Session()
            with sess.as_default():
                model = theme.Model(self.num_vocab, self.node_size, self.batch_size, self.embed_size, self.walk_length,
                                    self.sentence_len)
                opt = tf.train.AdamOptimizer(learning_rate=1e-3)
                train_op = opt.minimize(model.loss)
                sess.run(tf.global_variables_initializer())
                for epoch in range(self.epochs):
                    loss = 0
                    batches = self.dataset.generate_batches()
                    num_batch = len(batches)
                    for i in range(num_batch):
                        batch = batches[i]
                        node1, node2, node3 = zip(*batch)
                        node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                        text1, text2, text3 = self.text[node1], self.text[node2], self.text[node3]
                        # theme1, theme2, theme3 = self.theme[node1], self.theme[node2], self.theme[node3]
                        theme1 = []
                        theme2 = []
                        theme3 = []
                        for i in node1:
                            theme1.append(self.theme[i])
                        for i in node2:
                            theme2.append(self.theme[i])
                        for i in node3:
                            theme3.append(self.theme[i])

                        feed_dict = {
                            model.Text_a: text1,
                            model.Theme_a: theme1,
                            model.Text_b: text2,
                            model.Theme_b: theme2,
                            model.Text_neg: text3,
                            model.Theme_neg: theme3
                        }
                        _, loss_iter = sess.run([train_op, model.loss], feed_dict=feed_dict)
                        loss += loss_iter
                    print('epoch: ', epoch + 1, ' loss: ', loss)
                file = open('embedding.txt', 'wb')
                batches = self.dataset.generate_batches(mode='add')
                embed = [[] for _ in range(self.all_node)]
                for i in range(num_batch):
                    batch = batches[i]
                    node1, node2, node3 = zip(*batch)
                    node1, node2, node3 = np.array(node1), np.array(node2), np.array(node3)
                    text1, text2, text3 = self.text[node1], self.text[node2], self.text[node3]
                    # theme1, theme2, theme3 = self.theme[node1], self.theme[node2], self.theme[node3]
                    theme1 = []
                    theme2 = []
                    theme3 = []
                    for i in node1:
                        theme1.append(self.theme[i])
                    for i in node2:
                        theme2.append(self.theme[i])
                    for i in node3:
                        theme3.append(self.theme[i])
                    feed_dict = {
                        model.Text_a: text1,
                        model.Theme_a: theme1,
                        model.Text_b: text2,
                        model.Theme_b: theme2,
                        model.Text_neg: text3,
                        model.Theme_neg: theme3
                    }
                    A, B, Neg = sess.run([model.A, model.B, model.Neg], feed_dict=feed_dict)
                    for i in range(self.batch_size):
                        vector = list(A[i])
                        embed[node1[i]].append(vector)
                        vector = list(B[i])
                        embed[node2[i]].append(vector)

                for i in range(self.all_node):
                    # embed[i]  若节点i的出度为15 则len(embed[i])=15 embed[i]中有15个数组，每个数组是i和相连节点的特有的embedding
                    if embed[i]:
                        # print embed[i]
                        tmp = np.sum(embed[i], axis=0) / len(embed[i])  # 将一个节点的所有embedding相加 然后除以该节点的embedding数
                        file.write((' '.join(list(map(str, tmp))) + '\n').encode("utf-8"))
                    else:
                        file.write('\n'.encode("utf-8"))


if __name__ == '__main__':
    theme2vec = theme2vec()
    theme2vec.train()
