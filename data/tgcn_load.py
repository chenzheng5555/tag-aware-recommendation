from data.cf_load import CF_load
import data.utils as utils
from utility.word import CFG

import multiprocessing
from functools import partial
import torch
import numpy as np


class TGCN_load(CF_load):
    def __init__(self, args=None):
        super().__init__()
        self.cpu_core = CFG['cpu_core']
        self.args = args
        #--------------------------user-item-tag---------------------------------
        self.uit_data = utils.read_knowledge_data(self.file_dir, "user_item_tag.txt")
        uit_max = utils.column_info(self.uit_data)
        self.num['tag'] = uit_max[2] + 1
        #self.ui_adj = utils.to_sparse_adj(self.uit_data[:, 0], self.uit_data[:, 1], (self.num['user'], self.num['item']))
        self.ut_adj = utils.to_sparse_adj(self.uit_data[:, 0], self.uit_data[:, 2], (self.num['user'], self.num['tag']))
        self.it_adj = utils.to_sparse_adj(self.uit_data[:, 1], self.uit_data[:, 2], (self.num['item'], self.num['tag']))
        self.num['weight'] = int(max(self.ui_adj.max(), self.ut_adj.max(), self.it_adj.max()))

        print(f"TGCN_LOAD got ready! [{self.num}]")

    def get_sample_neighbor(self, neighbor_k):
        # ui ut iu it tu ti
        matrix = [self.ui_adj, self.ut_adj, self.ui_adj.transpose(),\
             self.it_adj, self.ut_adj.transpose(), self.it_adj.transpose()]
        sample = partial(utils.neighbor_sample, k=neighbor_k)
        if self.args.pool:
            results = self.args.pool.map(sample, matrix)
        else:
            pool = multiprocessing.Pool(self.cpu_core)
            results = pool.map(sample, matrix)
            pool.close()

        return results

    def get_all_neighbor(self):
        matrix = [self.ui_adj, self.ut_adj, self.ui_adj.transpose(),\
             self.it_adj, self.ut_adj.transpose(), self.it_adj.transpose()]
        max_deg = [max(adj.getnnz(1)) for adj in matrix]
        X = zip(matrix, max_deg)
        if self.args.pool:
            results = self.args.pool.map(utils.all_neighbor_sample, X)
        else:
            pool = multiprocessing.Pool(self.cpu_core)
            results = pool.map(utils.all_neighbor_sample, X)
            pool.close()

        return results

    def create_edge(self):
        edge_list = dict()
        # ui iu ut tu it ti
        user, item = self.ui_adj.row, self.ui_adj.col + self.num['user']
        edge_list[0] = np.stack([user, item])
        edge_list[1] = np.stack([item, user])

        user, tag = self.ut_adj.row, self.ut_adj.col + self.num['item'] + self.num['user']
        edge_list[2] = np.stack([user, tag])
        edge_list[3] = np.stack([tag, user])

        item, tag = self.it_adj.row + self.num['user'], self.it_adj.col + self.num['item'] + self.num['user']
        edge_list[4] = np.stack([item, tag])
        edge_list[5] = np.stack([tag, item])

        return edge_list
