
import time
import os
import numpy as np
import scipy.sparse as sp


#------------------------------file data--------------------------------------
def read_knowledge_data(file_dir, file_name):
    '''file data format:`entity_head relation entity_tail`\n
    or tag_assignment:`user item tag`
    '''
    start = time.time()
    file = os.path.join(file_dir, file_name)
    data = np.loadtxt(file, dtype=np.int32)
    uniq_data = np.unique(data, axis=0)

    print(f"got data from {file}, time spend:{(time.time()-start)/60:.2}")
    print(f"\tfile data info [repeat knowledge {len(data)-len(uniq_data)}]")
    return uniq_data


def read_interaction_data(file_dir, file_name):
    '''file data format:`u i1 i2 ...\n`'''
    start = time.time()
    u_items_dict = dict()
    rep_u, rep_i = 0, 0
    file = os.path.join(file_dir, file_name)

    with open(file, "r") as f:
        for line in f.readlines():
            data = [int(x) for x in line.strip().split(' ')]
            u, item = data[0], data[1:]
            items = list(set(item))
            if len(items) > 0:
                if u in u_items_dict.keys():
                    rep_u += 1
                    item = u_items_dict[u] + items
                    items = list(set(item))
                u_items_dict[u] = items

            rep_i += len(item) - len(items)

    print(f"got data from {file},time spend:{(time.time()-start)/60:.2}")
    print(f"\tfile data info [repeat user:{rep_u},repeat item:{rep_i}]")
    return u_items_dict


#----------------------------convert np dict sp----------------------------------------
def to_sparse_adj(row, col, shape):
    val = np.ones_like(row)
    adj = sp.coo_matrix((val, (row, col)), dtype=np.float32, shape=shape)
    return adj


def dict2np_array(data_dict):
    new_data = []
    for u, i in data_dict.items():
        ui = list(zip([u] * len(i), i))
        new_data.extend(ui)

    column_info(new_data)
    return np.array(new_data)


#------------------------------tgcn sample--------------------------------------
def neighbor_sample(matrix, k):
    '''
    对交互矩阵的`每一行`非0元素下标 `有放回地`采样K个,如果没有，则采样k个0\n
    return: [data,weight] 采样的邻域下标, 和对应的权重
    '''
    matrix = matrix.tocsr()
    data = np.zeros((matrix.shape[0], k), dtype=np.int)
    weight = np.zeros((matrix.shape[0], k), dtype=np.int)
    for i in range(matrix.shape[0]):
        nonzeroId = matrix[i].nonzero()[1]
        if len(nonzeroId):
            sampleId = np.random.choice(nonzeroId, k)
            # 将采样下标整体加一
            data[i] = sampleId + 1

            weight[i] = matrix[i].toarray()[0, sampleId]

    return [data, weight]


def all_neighbor_sample(X):
    matrix, max_deg = X
    matrix = matrix.tocsr()
    #max_deg = int(max(matrix.getnnz(1)))
    data = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    weight = np.zeros((matrix.shape[0], max_deg), dtype=np.int)
    for i in range(matrix.shape[0]):
        nonzeroId = matrix[i].nonzero()[1]
        x = len(nonzeroId)
        if x == 0:
            continue
        elif x < max_deg:
            sampleId = np.random.choice(nonzeroId, max_deg)
        else:
            sampleId = np.random.choice(nonzeroId, max_deg, replace=False)

        data[i] = sampleId + 1
        weight[i] = matrix[i].toarray()[0, sampleId]

    return [data, weight]

#----------------------------------------------------------------------------
def column_info(data_list):
    data = np.array(data_list)
    print(f"\t[{data.shape[0]}]:(unique,min,max)")
    max_list = []
    for i in range(data.shape[1]):
        col = data[:, i]
        max_list.append(max(col))
        print(f"\tcolumn {i}:[{len(set(col)),min(col),max(col)}]")

    return max_list


def dict_info(dict_data):
    user, item = [], []
    for u, i in dict_data.items():
        user.extend([u] * len(i))
        item.extend(list(i))
    data = np.stack([user, item]).T
    max_list = column_info(data)

    return data, max_list


#--------------------------------------------------------------------------

