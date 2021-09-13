import train_data.utils as utils
from train_data.abstract import Abstract_training_data

import numpy as np
import time
from utility.word import CFG
import torch
import multiprocessing
from functools import partial
import random

class BPR_training_data(Abstract_training_data):
    def __init__(self, data, args=None):
        super().__init__(args)
        self.batch_size = CFG['train_batch']
        self.num = data.num['item']
        self.num_user = data.num['user']
        self.train_ui = data.user_items['train']
        self.pos_inter = data.edge_index['train']
        self.args = args

        start = time.time()
        self.all_train_data = self.get_all_training_data()
        self.tot_inter = self.all_train_data.shape[0] // self.batch_size

        print(f"BPR_training_data producer tot_inter: {self.tot_inter},"\
            f"[all_training_data spend time:{time.time()-start}]")

    def get_all_training_data(self):

        list_inter = utils.split_data(self.pos_inter, self.cpu_core)
        sample = partial(utils.sample_neg_item, data_dict=self.train_ui, num_item=self.num)
        #--------------multi cpu-------------------
        if self.args.pool:
            results = self.args.pool.map(sample, list_inter)
        else:
            pool = multiprocessing.Pool(self.cpu_core)
            results = pool.map(sample, list_inter)
            pool.close()
        #----------------------------------

        data = np.vstack(results)
        data = utils.shuffle(data)
        data = torch.tensor(data, dtype=torch.long, device=self.device)
        return data

class DGCF_training_data(Abstract_training_data):
    '''sample method used in dgcf'''
    def __init__(self, data, args):
        super().__init__(args)
        self.batch_size = CFG['train_batch']
        self.cor_batch = CFG['cor_batch']
        self.num_item = data.num['item']
        self.num_user = data.num['user']
        self.num_tag = data.num['tag']
        self.train_ui = data.user_items['train']
        self.pos_inter = data.edge_index['train']

        self.tot_inter = self.pos_inter.shape[0] // self.batch_size + 1
        #self.cor_batch = int(max(self.num_user, self.num_item) / self.tot_inter)
        start = time.time()
        self.mini_sample()
        print(f"DGCF_training_data producer,tot_inter:{self.tot_inter},cor_batch:{self.cor_batch}," \
            f"[mini_sample time:{time.time()-start}]")

    def mini_sample(self):
        data = utils.ngcf_sample(self.train_ui, self.batch_size, self.num_item)
        data = torch.tensor(data, dtype=torch.long, device=self.device)
        cor_user = random.sample(list(range(self.num_user)), self.cor_batch)
        cor_item = random.sample(list(range(self.num_item)), self.cor_batch)
        if CFG['use_tag']:
            cor_tag = random.sample(list(range(self.num_tag)), self.cor_batch)
            cor = np.stack([cor_user, cor_item, cor_tag])
        else:
            cor = np.stack([cor_user, cor_item])
        cor = torch.tensor(cor, dtype=torch.long, device=self.device)
        return data, cor

    def reset(self):
        pass

    def mini_batch(self):
        for _ in range(0, self.tot_inter):
            yield self.mini_sample()
