from math import pow


def InitNegTable(edges):
    a_list, b_list = zip(*edges)
    a_list = list(a_list)
    b_list = list(b_list)
    NEG_SAMPLE_POWER = 0.75
    node = a_list
    node.extend(b_list)

    node_degree = {}
    for i in node:
        if i in node_degree:
            node_degree[i] += 1
        else:
            node_degree[i] = 1
    sum_degree = 0
    for i in node_degree.values():
        sum_degree += pow(i, 0.75)

    por = 0
    cur_sum = 0
    vid = -1
    neg_table = []
    degree_list = list(node_degree.values())
    node_id = node_degree.keys()
    for i in range(1000000):
        if (((i + 1) / float(1000000)) > por):
            cur_sum += pow(degree_list[vid + 1], NEG_SAMPLE_POWER)
            por = cur_sum / sum_degree
            vid += 1
        neg_table.append(list(node_id)[vid])
    print(len(neg_table))
    return neg_table
