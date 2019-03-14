import random

locat = "../../data/hepth/"
graph_path = locat + 'graph.txt'

f = open(graph_path, 'rb')
edges = [i for i in f]
selected = random.sample(edges, int(len(edges) * 0.05))
remain = [i for i in edges if i not in selected]
fw1 = open('graph_test.txt', 'wb')
fw2 = open('graph_train.txt', 'wb')

for i in selected:
    fw1.write(i)
for i in remain:
    fw2.write(i)
