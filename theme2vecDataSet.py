import sys
import random
from tensorflow.contrib import learn
import numpy as np
from negativeSample import InitNegTable
import tqdm
import collections


class theme2vecDataSet:
    def __init__(self, text_path, graph_path, batch_size,sentence_len):
        text_file, graph_file = self.load(text_path, graph_path)
        self.graph_path = graph_path
        self.sentence_len = sentence_len
        self.edges = self.load_edges(graph_file)
        self.text, self.num_vocab, self.num_nodes = self.load_text(text_file)
        self.batch_size = batch_size
        self.negative_table = InitNegTable(self.edges)

    def load(self, text_path, graph_path):
        text_file = open(text_path, 'rb').readlines()
        graph_file = open(graph_path, 'rb').readlines()

        return text_file, graph_file

    def load_edges(self, graph_file):
        edges = []
        for i in graph_file:
            edges.append(list(map(int, i.strip().decode("utf-8").split('\t'))))
        return edges

    def load_text(self, text_file):
        vocab = learn.preprocessing.VocabularyProcessor(self.sentence_len)
        texts = []
        for i in text_file:
            texts.append(i.decode("utf-8"))
        text = np.array(list(vocab.fit_transform(texts)))
        num_vocab = len(vocab.vocabulary_)
        num_nodes = len(text)

        return text, num_vocab, num_nodes

    def generate_batches(self, mode=None):
        num_batch = len(self.edges) // self.batch_size
        edges = self.edges
        if mode == 'add':
            num_batch += 1
            edges.extend(edges[:(self.batch_size - len(self.edges) // self.batch_size)])
        if mode != 'add':
            random.shuffle(edges)
        sample_edges = edges[:num_batch * self.batch_size]
        sample_edges = self.negative_sample(sample_edges)
        # print(sample_edges)
        batches = []
        for i in range(num_batch):
            batches.append(sample_edges[i * self.batch_size:(i + 1) * self.batch_size])
        return batches

    def negative_sample(self, edges):
        node1, node2 = zip(*edges)
        sample_edges = []
        func = lambda: self.negative_table[random.randint(0, 1000000 - 1)]
        for i in range(len(edges)):
            neg_node = func()
            while node1[i] == neg_node or node2[i] == neg_node:
                neg_node = func()
            sample_edges.append([node1[i], node2[i], neg_node])

        return sample_edges

    ###### Build Trees #########
    def str_list_to_int(self, str_list):
        return [int(item) for item in str_list]

    def read_edges_from_file(self):
        with open(self.graph_path, "r") as f:
            lines = f.readlines()
            edges = [self.str_list_to_int(line.split()) for line in lines]
            return edges

    def read_edges(self):
        graph = {}
        nodes = set()
        train_edges = self.read_edges_from_file()

        for edge in train_edges:
            nodes.add(edge[0])
            nodes.add(edge[1])
            if graph.get(edge[0]) is None:
                graph[edge[0]] = []
            if graph.get(edge[1]) is None:
                graph[edge[1]] = []
            graph[edge[0]].append(edge[1])
            graph[edge[1]].append(edge[0])

        return nodes, len(nodes), graph

    def buildTree(self, nodes, graph):
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees
