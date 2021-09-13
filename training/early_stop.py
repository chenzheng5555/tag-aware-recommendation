from typing import Iterable
from utility.word import CFG

import torch


class Early_stop:
    def __init__(self, args):
        self.best_value = None
        self.count_step = 0
        self.best_result = None
        self.best_epoch = 0

        self.patient_step = CFG['patient_epoch']
        self.save_path = f"{args.out_dir}/model.pth.tar"
        self.key = CFG['early_stop_key']

        if self.key in ['precision', 'recall', 'ndcg']:
            self.cmp = lambda x, y: x > y
        else:
            self.cmp = lambda x, y: x < y

    def __call__(self, model, cur_results, epoch):
        if isinstance(cur_results[self.key], Iterable):
            cur_res = cur_results[self.key][0]
        else:
            cur_res = cur_results[self.key]

        if self.best_value == None or self.cmp(cur_res, self.best_value):
            self.best_value = cur_res
            self.count_step = 0
            # torch.save(model, self.save_path)
            torch.save(model.state_dict(), self.save_path)
            self.best_result = cur_results
            self.best_epoch = epoch
        else:
            self.count_step += 1

        if self.count_step > self.patient_step:
            return True
        return False
