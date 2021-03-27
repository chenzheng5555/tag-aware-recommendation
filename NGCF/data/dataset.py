from torch.utils.data import Dataset
import numpy as np
from .utils import printc, neg_sample
import time


class TransTag(Dataset):
    def __init__(self, data, sample_k=1):
        self.data = []
        t_u, t_i = data.transTag
        for t in t_u.keys():
            k = np.random.randint(0, 3)
            if k == 0:
                for i in t_i[t]:
                    pos_u = list(t_u[t])
                    pos = list(zip(list(np.repeat(pos_u, sample_k)), [t] * len(pos_u) * sample_k, [i] * len(pos_u) * sample_k))
                    neg_u = neg_sample(pos_u, sample_k, data.num_user)
                    neg = list(zip(neg_u, [t] * len(pos_u) * sample_k, [i] * len(pos_u) * sample_k))
                    self.data.extend(list(zip(pos, neg)))
            elif k == 1:
                for u in t_u[t]:
                    pos_i = list(t_i[t])
                    pos = list(zip([u] * len(pos_i) * sample_k, [t] * len(pos_u) * sample_k, list(np.repeat(pos_i, sample_k))))
                    neg_i = neg_sample(pos_i, sample_k, data.num_item)
                    neg = list(zip([u] * len(pos_u) * sample_k), [t] * len(pos_u) * sample_k, neg_i)
                    self.data.extend(list(zip(pos, neg)))
            elif k == 2:
                pos_u = list(t_u[t])
                pos_i = list(t_i[t])
                pos = list(zip(list(np.repeat(pos_u, len(pos_i))), [t] * len(pos_u) * len(pos_i), list(np.repeat(pos_i, len(pos_u)))))
                neg_u = neg_sample(pos_u, sample_k, data.num_user)
                neg_i = neg_sample(pos_i, sample_k, data.num_item)
                neg = list(zip(list(np.repeat(neg_u, len(neg_i))), [t] * len(neg_u) * len(neg_i), list(np.repeat(neg_i, len(neg_u)))))
                self.data.extend(list(zip(pos, neg)))
        printc(f"transTag data size:{len(self.data)}")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TrainSet(Dataset):
    def __init__(self, data, sample_k=1):
        self.data = []
        start = time.time()
        for u in data.train_set.keys():
            pos = data.train_set[u]
            neg = neg_sample(pos, sample_k, data.num_item)
            self.data.extend(list(zip([u] * len(pos) * sample_k, list(np.repeat(pos, sample_k)), neg)))
        printc(f"train data size:{len(self.data)},time spend:{time.time()-start}")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class TestSet(Dataset):
    def __init__(self, data):
        self.user = list(data.test_set.keys())
        # for u in data.test_set.keys():
        #     self.user.append(u)
        printc(f"test data size:{len(self.user)}")

    def __getitem__(self, index):
        return self.user[index]

    def __len__(self):
        return len(self.user)
