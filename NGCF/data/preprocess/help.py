import numpy as np
import random
from collections import defaultdict


#-------------------------输出文件-----------------------------------
def write_dict(data_dict, out_file):
    with open(out_file, "w") as fp:
        for u, items in data_dict.items():
            s = f"{u}"
            for i in items:
                s += f" {i}"
            fp.write(s + '\n')


def write_list(data_list, out_file):
    with open(out_file, "w") as fp:
        for i, d in enumerate(data_list):
            fp.write(f"{i} {d}\n")


#-------------------------读取文件-----------------------------------
def get_file_data(file_path, start, end):
    data_list = []
    with open(file_path, mode='r') as fp:
        for line in fp.readlines()[start:]:
            data = line.strip('\n').split('\t')
            data = list(map(int, data[:end]))
            data_list.append(data)

    print(f"get user's tag assignment from [{file_path}]")
    return np.array(data_list)


#-------------------------数据变换-----------------------------------
def get_dict_from_list(data_list, key, value):
    data_dict = defaultdict(set)
    for data in data_list:
        k, v = data[key], data[value]
        data_dict[k].add(v)
    return data_dict


def get_train_UIT(train_dict, data_list):
    train_data = []
    for data in data_list:
        u, i = data[0], data[1]
        #只有后面的判断，会修改train——dict
        if u in train_dict.keys() and i in train_dict[u]:
            train_data.append(data)

    return np.array(train_data)


#-------------------------数据处理-----------------------------------
def index_to_dense(data_list, file_path=None):
    unique_data, new_data = np.unique(data_list, return_inverse=True)
    if file_path != None:
        write_list(unique_data, file_path)
        print(f"make the index to dense and save the maps to [{file_path}]")
    return new_data


def delete_tag(data_list, least_k):
    tag_set = defaultdict(int)
    tags = np.array(data_list)[:, 2]
    for t in tags:
        tag_set[t] += 1
    new_data = []
    for data in data_list:
        if tag_set[data[2]] >= least_k:
            new_data.append(data)

    print(f"delete tag appear less than [{least_k}] times")
    return np.array(new_data)


#-------------------------数据划分-----------------------------------
def random_split_user_items_dict(user_items, train_ratio, val_ratio=0, off=0):
    r"""
    以比率随机划分数据，以 每个用户的交互 为基础，如果只有一个item， 则划分到测试集
    """
    train_user_items, test_user_items = defaultdict(list), defaultdict(list)
    val_user_items = defaultdict(list)

    for u, items in user_items.items():
        if len(items) == 0:
            continue
        items = set(items)
        k = int(len(items) * train_ratio + off)
        if k > 0:
            train = random.sample(items, k)
            train_user_items[u].extend(train)
            items = items - set(train)
            if len(items) == 0:
                continue

        if val_ratio != 0:
            k = int(len(items) * val_ratio)
            if k != 0:
                val = random.sample(items, k)
                val_user_items[u].extend(val)
                items = items - set(val)
                if len(items) == 0:
                    continue
        
        test_user_items[u].extend(items)
    
    print(f"split train,val set ratio:[{train_ratio,val_ratio}],training off {off}")
    if val_ratio != 0:
        return train_user_items, test_user_items, val_user_items
    return train_user_items, test_user_items


def change_dict(train_dict, test_dict):
    '''交互训练集和测试集，以便所有user和item都在train set出现过'''
    cnt_user, cnt_item, cnt_del_user = 0, 0, 0
    all_train_item = defaultdict(int)
    for items in train_dict.values():
        for i in items:
            all_train_item[i] += 1

    for u in test_dict.keys():
        if u not in train_dict.keys():
            train_dict[u] = test_dict[u]
            i = test_dict[u][0]
            all_train_item[i] += 1
            test_dict.pop(u)
            cnt_user += 1
            continue

        to_test, del_test = [], []
        for item in test_dict[u]:
            if item in all_train_item.keys():
                continue

            cnt_item += 1
            found = False
            for i in train_dict[u]:
                if all_train_item[i] > 1:
                    all_train_item[i] -= 1
                    all_train_item[item] += 1
                    if isinstance(train_dict[u], list):
                        train_dict[u].append(item)
                    elif isinstance(train_dict[u], set):
                        train_dict[u].add(item)
                    train_dict[u].remove(i)
                    to_test.append(i)
                    del_test.append(item)
                    found = True
                    break

            if found == False:
                train_dict[u].append(item)
                all_train_item[item] += 1
                del_test.append(item)

        if isinstance(train_dict[u], list):
            test_dict[u].extend(to_test)
        elif isinstance(train_dict[u], set):
            test_dict[u].update(to_test)
        
        for i in del_test:
            test_dict[u].remove(i)
        if len(test_dict[u]) == 0:
            cnt_del_user += 1
            test_dict.pop(u)

    print(f"num user,del_user,item changed:{cnt_del_user,cnt_user,cnt_item}")