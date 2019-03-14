import random
import numpy as np

node2vec = dict()
with open('../embedding.txt') as f:
    for idx, line in enumerate(f.readlines()):
        nodes = [float(i) for i in line.strip().split()]
        node2vec[idx] = nodes
# f = open('embed_hep.txt', 'rb')
# for i, j in enumerate(f):  # i:index j:item
#     if j != '\n':
#         node2vec[i] = map(float, j.strip().decode("utf-8").split(' '))
print(len(node2vec))  # node2vec 为节点的表示举证
with open('../graph_test.txt') as f:
    edges = [[int(j) for j in i.strip().split()] for i in f]  # 测试集中的边

nodes = list(set([i for j in edges for i in j]))  # 节点 2331 遍历edge中的每个元组j, 再遍历每个j，然后去重。 nodes为测试集中的节点数
a = 0
b = 0
for i, j in edges:
    if len(node2vec[i]) != 0 and len(node2vec[j]) != 0:
        dot1 = np.dot(node2vec[i], node2vec[j])  # 点乘测试集中边上两个节点(i,j)的表示向量
        random_node = random.sample(nodes, 1)[0]  # 从序列nodes中取1个随机且独立的元素
        # 当随机选取的节点为j 或者 随机节点不在node2vec中时重新选取节点
        while random_node == j or len(node2vec[random_node]) == 0:
            random_node = random.sample(nodes, 1)[0]
        dot2 = np.dot(node2vec[i], node2vec[random_node])
        if dot1 > dot2:
            a += 1
        elif dot1 == dot2:
            a += 0.5
        b += 1

print(float(a) / b)
