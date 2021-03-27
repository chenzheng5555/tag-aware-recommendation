import numpy as np
from collections import defaultdict
import random
import os


def printc(values="", c='=', k=5):
    print(f"{c*k}", end="")
    print(values)


def neg_sample(pos, sam_k, len_pos):
    '''
    from range(0,len_pos) sample sam_k*len(pos) integers that not in pos 
    '''
    neg = []
    for _ in pos:
        for _ in range(sam_k):
            while True:
                j = np.random.randint(0, len_pos)
                if j not in pos:
                    neg.append(j)
                    break
    return neg


def check(dic, ke, cnt):
    if ke in dic.keys():
        return cnt
    else:
        dic[ke] = cnt
        cnt += 1
        return cnt


def write_file(file_name, dic):
    with open(file_name, 'w') as f:
        for k, v in dic.items():
            f.write(str(k) + ' ' + str(v) + '\n')


def deal_index_fuc(user_item, tagging=None, out_txt=False):
    '''
    delete the index that has never appear,only can deal the dict and list type.\n
    dict: key and values are different class\n
    list: each column is a class
    '''
    printc('deal the index')

    cnt = [0, 0, 0]
    maps = [dict() for _ in range(3)]
    new_user_item = defaultdict(list)
    for k in user_item.keys():
        cnt[0] = check(maps[0], k, cnt[0])
        for i in user_item[k]:
            cnt[1] = check(maps[1], i, cnt[1])
            new_user_item[maps[0][k]].append(maps[1][i])

    if tagging != None:
        new_tagging = []
        for line in tagging:
            for i, k in enumerate(line):
                cnt[i] = check(maps[i], k, cnt[i])
            new_tagging.append([maps[i][k] for i, k in enumerate(line)])
        tagging = new_tagging

    printc(f"done, the lens of maps are: {[len(m) for m in maps]}")

    if out_txt:
        for i, m in enumerate(maps):
            write_file("map_file_%d_txt" % i, m)

    return new_user_item, tagging, [len(m) for m in maps]


def data_split(user_items, train_rate, valid_rate=None):
    '''
    using user_items data to split trainset,validset,testset.\n
    user_items: is dict type
    '''
    printc("split train, valid, test data from all items of each user")
    test_set = dict()
    train_set = dict()
    valid_set = dict()
    user_no_train = []
    for u in user_items.keys():
        train = random.sample(list(user_items[u]), int(len(user_items[u]) * train_rate))
        if len(train) == 0:
            user_no_train.append(u)
        else:
            train_set[u] = train
        test = set(user_items[u]).difference(set(train))
        if valid_rate == None:
            test_set[u] = list(test)
            continue
        valid = random.sample(list(test), int(len(user_items[u]) * valid_rate))
        if len(valid) != 0:
            valid_set[u] = valid
        test_set[u] = list(test.difference(set(valid)))

    printc(f"number user no train data:{[len(user_no_train)]}")
    return train_set, test_set, valid_set


def get_edge(train_set, num_user, num_item, tagging_data=None):
    '''
    use train_set data and taggging_data if given to get the graph
    return edge_index,weight,transTag
    '''
    printc("constract the graph edge_index from train_set")
    edge_index = set()
    weight = defaultdict(int)
    u_i = defaultdict(set)
    user_tag = defaultdict(set)  # the items that user u assign tag t
    item_tag = defaultdict(set)  # the users who assign tag t to item i

    for u in train_set.keys():
        for i in train_set[u]:
            ii = i + num_user
            weight[(u, ii)] = weight[(ii, u)] = 1
            edge_index.add((u, ii))
            edge_index.add((ii, u))

    if tagging_data == None:
        return edge_index, None, None

    for tagging in tagging_data:
        u, i, t = tagging
        u_i[(u, i)].add(t)
    for u in train_set.keys():
        for i in train_set[u]:
            if (u, i) in u_i.keys():
                for t in u_i[(u, i)]:
                    tt = t + num_user + num_item
                    weight[(tt, u)] = weight[(u, tt)] = weight[(u, tt)] + 1
                    weight[(tt, ii)] = weight[(ii, tt)] = weight[(ii, tt)] + 1
                    edge_index.add((u, tt))
                    edge_index.add((tt, u))
                    edge_index.add((ii, tt))
                    edge_index.add((tt, ii))
                    user_tag[(u, t)].add(i)
                    item_tag[(i, t)].add(u)

    return edge_index, weight, (user_tag, item_tag)  # construct transTag


def data_statistic(data):
    '''
    statistic of the data(list or dict),return [max_user+1,max_item+1,(max_tag)+1]
    '''
    user, item = [], []
    cnt = 0
    if isinstance(data, defaultdict):
        rep = dict()
        for u in data.keys():
            items = data[u]
            if len(items) != len(set(items)):
                rep[u] = len(items) - len(set(items))
                data[u] = list(set(items))  # delete repeat item
            cnt += len(data[u])
            user.append(u)
            item.extend(items)
        if len(rep) != 0:
            printc(f"the number of user appear repeat interaction:{len(rep)}")
        mlm = [user, item]

        max_node = [max(i) + 1 for i in mlm]

    elif isinstance(data, list):
        data = np.array(data)
        mlm = [list(data[:, 0]), list(data[:, 1]), list(data[:, 2])]
        l = len(data)
        s = list(zip(mlm[0], mlm[1], mlm[2]))
        if len(s) != len(set(s)):
            printc(f"repeat assignment:{len(s)-len(set(s))}")
            l = len(set(s))
        cnt += l
        max_node = [max(i) + 1 for i in mlm]

    printc(f"number interaction:{cnt}")
    printc("(max len(set) min) statistic:")
    printc(f"{[(max(i),len(set(i)),min(i)) for i in mlm]}")

    return max_node


def user_interaction(file_path, file_name):
    '''
    every line is one user' interactions, [u,i,...,i]
    '''
    printc(f"{file_name}-user_interaction")
    user_item = defaultdict(list)
    with open(os.path.join(file_path, file_name), 'r') as f:
        for line in f.readlines():
            l = line.strip('\n').split(' ')
            u = int(l[0])
            i = [int(i) for i in l[1:]]
            user_item[u].extend(i)

    max_node = data_statistic(user_item)

    return user_item, max_node


def user_assignment(file_path, file_name, least=0, use_tag=False):
    '''
    evert line is just one interaction/assignment, [u,i,t]\n
    least: delete the tags that appear times less than least
    '''
    printc(f"{file_name}-user_assignment")
    tag_count = defaultdict(int)
    data = list()
    user_item = defaultdict(set)
    with open(os.path.join(file_path, file_name), 'r') as f:
        for line in f.readlines()[1:]:
            u, i, t = list(map(int, line.strip('\n').split('\t')[:3]))
            tag_count[t] += 1
            data.append([u, i, t])
            user_item[u].add(i)
        max_node = data_statistic(data)

    new_data = list()
    if least > 0:
        user_item = defaultdict(set)
        cnt = 0
        for x in data:
            u, i, t = x
            if tag_count[t] < least:
                cnt += 1
                continue
            new_data.append(x)
            user_item[u].add(i)
        data = new_data
        printc(f"delete {cnt} assignments which tag appear times less than {least}")
        max_node = data_statistic(data)

    if use_tag == False:
        data = None

    return user_item, data, max_node
