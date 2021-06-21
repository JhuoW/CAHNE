import random
import numpy as np

node2vec = dict()
with open('../embedding.txt') as f:
    for idx, line in enumerate(f.readlines()):
        nodes = [float(i) for i in line.strip().split()]
        node2vec[idx] = nodes

print(len(node2vec)) 
with open('../graph_test.txt') as f:
    edges = [[int(j) for j in i.strip().split()] for i in f]  

nodes = list(set([i for j in edges for i in j])) 
a = 0
b = 0
for i, j in edges:
    if len(node2vec[i]) != 0 and len(node2vec[j]) != 0:
        dot1 = np.dot(node2vec[i], node2vec[j]) 
        random_node = random.sample(nodes, 1)[0] 
        while random_node == j or len(node2vec[random_node]) == 0:
            random_node = random.sample(nodes, 1)[0]
        dot2 = np.dot(node2vec[i], node2vec[random_node])
        if dot1 > dot2:
            a += 1
        elif dot1 == dot2:
            a += 0.5
        b += 1

print(float(a) / b)
