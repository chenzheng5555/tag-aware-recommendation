from train_data.abstract import Abstract_training_data
import train_data.utils as utils
from utility.word import CFG

import time
import torch
import numpy as np
from functools import partial
import multiprocessing


class KGAT_training_data(Abstract_training_data):
    def __init__(self, data, args):
        super().__init__(args)
        self.batch_size = CFG['transe_batch']
        self.num = data.num['user'] + data.num['item'] + data.num['tag']
        edge_list = data.create_edge()
        kg_data = []
        for k in edge_list.keys():
            kg_data.append(np.vstack([edge_list[k], np.ones(edge_list[k].shape[1]) * k]))

        self.all_triplet = np.hstack(kg_data).transpose()[:, [0, 2, 1]]

        self.h_r_dict = utils.get_h_r_dict(self.all_triplet)
        self.tot_inter = self.all_triplet.shape[0] // self.batch_size
        start = time.time()
        self.mini_batch()
        print(f"TransE_training_data producer, tot_inter: {self.tot_inter},"\
            f"[mini_sample time:{time.time()-start}]")

    def reset(self):
        pass

    def mini_batch(self):
        for i in range(0, self.tot_inter):
            batch = self.all_triplet[i:i + self.batch_size]
            data = utils.sample_neg_tail(batch, self.h_r_dict, self.num)
            data = torch.tensor(data, dtype=torch.long, device=self.device)
            yield data


class TransTag_training_data(Abstract_training_data):
    def __init__(self, data, args=None):
        super().__init__(args)
        self.args = args
        self.batch_size = CFG['transtag_batch']

        self.num = data.num['item']
        self.uti_data = data.uit_data[:, [0, 2, 1]]
        self.u_t_dict = utils.get_h_r_dict(self.uti_data)
        start = time.time()
        self.all_train_data = self.get_all_training_data()
        self.tot_inter = self.all_train_data.shape[0] // self.batch_size
        print(f"TransTag_training_data producer, tot_inter: {self.tot_inter},"\
            f"[all_training_data time:{time.time()-start}]")

    def get_all_training_data(self):
        list_tri = utils.split_data(self.uti_data, self.cpu_core)
        sample = partial(utils.sample_neg_tail, data_dict=self.u_t_dict, num=self.num)
        #--------------multi cpu-------------------
        if self.args.pool:
            results = self.args.pool.map(sample, list_tri)
        else:
            pool = multiprocessing.Pool(self.cpu_core)
            results = pool.map(sample, list_tri)
            pool.close()
        #----------------------------------

        data = np.vstack(results)
        data = torch.tensor(data, dtype=torch.long, device=self.device)
        return data