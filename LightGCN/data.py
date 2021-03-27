import torch
import os
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict


class DataBase():
    def __init__(self, config):
        self.path = config.path
        self.neg_sample_k = config.neg_sample_k
        self.device = config.device
        self.num_items, self.num_users = 0, 0

        self.trainUser, self.trainItem, self.userPos = self.getData("train.txt")
        _, _, self.userTest = self.getData("test.txt")
        self.num_items += 1
        self.num_users += 1
        print(f"num_users:{self.num_users},num_items:{self.num_items}")

    def getData(self, filename):
        user, item = [], []
        user_dict = defaultdict(list)
        with open(os.path.join(self.path, filename)) as f:
            for l in f.readlines():
                if len(l) == 0:
                    continue
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                u = int(l[0])
                self.num_users = max(self.num_users, u)
                self.num_items = max(self.num_items, max(items))
                user_dict[u].extend(items)
                user.extend([u] * len(items))
                item.extend(items)
        print(f"[{filename}]:iteractions-{len(item)}")

        return np.array(user), np.array(item), user_dict

    # 有重复的边，会使节点的度增加
    def getEdges(self):
        item = self.trainItem + self.num_users
        user = self.trainUser
        source = np.append(user, item)
        target = np.append(item, user)
        edge_index = torch.tensor([source, target], dtype=torch.long)
        edge_index = edge_index.to(self.device)
        return edge_index

    # 为一个用户所有的正样本采样k个负样本
    def sample_neg_k(self, u, pos_items, k):
        data = []
        i = np.random.randint(0, len(pos_items))
        i = pos_items[i]
        #for i in pos_items:
        for _ in range(k):
            while True:
                j = np.random.randint(0, self.num_items)
                if j not in pos_items:
                    data.append([u, i, j])
                    break
        return np.array(data)

    def getTrainData(self):
        TrainData = []
        for u, items in self.userPos.items():
            if len(items)==0:
                continue
            data = self.sample_neg_k(u, items, k=self.neg_sample_k)
            TrainData.append(data)
        data = np.vstack(TrainData)
        data = torch.tensor(data, dtype=torch.long, device=self.device)
        return data


class TrainSet(Dataset):
    def __init__(self, data):
        self.data = data.getTrainData()

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
