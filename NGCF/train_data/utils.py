import numpy as np
from collections import defaultdict
import random
#------------------------------------------------------
def split_data(np_data, k):
    size = len(np_data) // k
    list_data = []
    for i in range(k):
        start = i * size
        if i == k - 1:
            end = len(np_data)
        else:
            end = (i + 1) * size
        list_data.append(np_data[start:end])

    return list_data


def sample_neg_item(pos_inter, data_dict, num_item):
    data = []
    for u, pos_i in pos_inter:
        while True:
            idx = np.random.randint(0, num_item)
            if idx not in data_dict[u]:
                data.append([u, pos_i, idx])
                break

    return np.array(data)


def sample_neg_tail(pos_inter, data_dict, num):
    data = []
    for h, r, t in pos_inter:
        while True:
            idx = np.random.randint(0, num)
            if idx not in data_dict[h][r]:
                data.append([h, r, t, idx])
                break

    return np.array(data)


def get_h_r_dict(all_triplet):
    head_dict = defaultdict(dict)
    for h, r, t in all_triplet:
        if r not in head_dict[h]:
            head_dict[h][r] = []
        head_dict[h][r].append(t)
    return head_dict


def shuffle(data):
    indexs = np.arange(len(data))
    np.random.shuffle(indexs)
    return data[indexs]


def ngcf_sample(true_ui, batch, num_item):
    all_user = list(true_ui.keys())
    if len(all_user) > batch:
        sample_user = random.sample(all_user, batch)
    else:
        sample_user = np.random.choice(all_user, batch)

    data = []
    for u in sample_user:
        pos_i = np.random.choice(true_ui[u])
        neg_i = sample_one_neg(true_ui[u], num_item)
        data.append([u, pos_i, neg_i])

    return np.array(data)


def sample_one_neg(pos_list, num):
    while True:
        idx = np.random.randint(0, num)
        if idx not in pos_list:
            return idx