from data.cf_load import CF_load
import data.utils as utils
from utility.word import CFG

import numpy as np


class KGAT_load(CF_load):
    def __init__(self, args = None):
        super().__init__(args)
        #--------kg_data------------
        self.kg_data = utils.read_knowledge_data(self.file_dir, "kg_final.txt")
        hrt_max = utils.column_info(self.kg_data)
        self.num['entity'] = max(hrt_max[0], hrt_max[2]) + 1
        self.num_rela = hrt_max[1] + 1
        self.num['relation'] = (self.num_rela + 1) * 2

        self.all_triplet = self.get_all_triplet()

        print(f"KGAT_load got ready!!! [{self.num}]")

    def get_all_triplet(self):
        user = self.edge_index['train'][:, 0]
        item = self.edge_index['train'][:, 1] + self.num['user']
        ui_rela = np.vstack([user, np.ones_like(user) * 0, item])
        ui_rela = ui_rela.transpose()
        r_ui_rela = ui_rela[:, [2, 1, 0]]
        r_ui_rela[:, 1] += self.num_rela + 1

        head = self.kg_data[:, 0] + self.num['user']
        rela = self.kg_data[:, 1] + 1
        tail = self.kg_data[:, 2] + self.num['user']
        kg_rela = np.vstack([head, rela, tail])
        kg_rela = kg_rela.transpose()
        r_kg_rela = kg_rela[:, [2, 1, 0]]
        r_kg_rela[:, 1] += self.num_rela + 1
        all_rela = np.vstack([ui_rela, r_ui_rela, kg_rela, r_kg_rela])
        return all_rela

    def get_relation_dict(self):

        # sort
        # id = np.lexsort((all_rela[:, 2], all_rela[:, 1], all_rela[:, 0]))
        # all_rela = all_rela[id]

        all_kg_dict = dict()
        for k in range(self.num['relation']):
            all_kg_dict[k] = self.all_triplet[self.all_triplet[:, 1] == k][:, [0, 2]]

        return all_kg_dict
