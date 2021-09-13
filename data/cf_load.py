import data.utils as utils
from utility.word import CFG

import os
import numpy as np


class CF_load(object):
    def __init__(self, args = None):
        self.file_dir = os.path.join(CFG['data_root'], CFG['dataset'])
        self.has_val = CFG['has_val']
        #---------------read train.txt\test.txt\val.txt----------------------------
        self.user_items, self.num, self.edge_index = dict(), dict(), dict()
        max_dict = dict()
        self.user_items['train'] = utils.read_interaction_data(self.file_dir, "train.txt")
        self.edge_index['train'], max_dict['train'] = utils.dict_info(self.user_items['train'])
        if self.has_val == True:
            self.user_items['val'] = utils.read_interaction_data(self.file_dir, "val.txt")
            self.edge_index['val'], max_dict['val'] = utils.dict_info(self.user_items['val'])
        self.user_items['test'] = utils.read_interaction_data(self.file_dir, "test.txt")
        self.edge_index['test'], max_dict['test'] = utils.dict_info(self.user_items['test'])
        #----------------get the num_user\num_item----------
        self.num['user'], self.num['item'] = np.array(list(max_dict.values())).max(axis=0) + 1
        #----------------get user-item-adj----------
        self.ui_adj = utils.to_sparse_adj(self.edge_index['train'][:, 0], \
            self.edge_index['train'][:, 1], (self.num['user'], self.num['item']))

        print(f"CF_LOAD got ready! [{self.num}]")
