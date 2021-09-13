import training.utils as utils
from utility.word import CFG

import torch
import numpy as np
from functools import partial
from collections import defaultdict
import multiprocessing

#print("-----------basic_test---------------")

def test_one_batch(X):
    return test_users(X[1], X[0])


def test_users(true_ui, rating):
    assert len(true_ui) == len(rating)
    label = utils.get_label(true_ui, rating)
    result = defaultdict(list)
    for k in CFG['topks']:
        ret = utils.pre_rec_k(label, true_ui, k)
        for key in ret.keys():
            result[key].append(ret[key])
        ndcg = utils.ndcg_k(label, true_ui, k)
        result['ndcg'].append(ndcg)

    return result


def epoch_test(model, pos_ui, true_ui, args, all_users=None):
    rating_list, true_list, auc = [], [], []
    if all_users == None:
        all_users = list(true_ui.keys())
    max_k = max(CFG['topks'])
    with torch.no_grad():
        for user in utils.minibatch(all_users, CFG['test_batch']):
            allpos = [pos_ui[u] if u in pos_ui.keys() else [] for u in user]
            groundTrue = [true_ui[u] for u in user]
            gpu_user = torch.tensor(user, dtype=torch.long, device=CFG['device'])
            rating = model.predict_rating(gpu_user)

            del_pos_row, del_pos_col = [], []
            for i, item in enumerate(allpos):
                del_pos_row.extend([i] * len(item))
                del_pos_col.extend(item)

            rating[del_pos_row, del_pos_col] = -(1 << 10)
            _, topk_rating = torch.topk(rating, k=max_k)

            rating_list.append(topk_rating.cpu().numpy())
            true_list.append(groundTrue)
            for i, test_data in enumerate(groundTrue):
                auc.append(utils.auc(rating[i].detach().cpu().numpy(), rating.shape[1], test_data))

            #print(test_users(groundTrue, topk_rating))   #测试

    #-----------------mult cpu------------------
    assert len(auc) == len(all_users)

    X = zip(rating_list, true_list)
    if args.pool:
        results = args.pool.map(test_one_batch, X)
    else:
        pool = multiprocessing.Pool(CFG['cpu_core'])
        results = pool.map(test_one_batch, X)
        pool.close()

    #-----------------result------------------
    tot_ret = defaultdict(list)
    for ret in results:
        for key in ret.keys():
            tot_ret[key].append(ret[key])
    tot_ret['auc'].append([np.sum(auc)])

    ret = dict()
    for key in tot_ret.keys():
        ret[key] = list(np.sum(np.array(tot_ret[key]), axis=0) / len(all_users))


    return ret


class Basic_test():
    def __init__(self, data, args=None):
        self.args = args
        self.pos_ui = data.user_items['train']
        self.true_ui = dict()
        if CFG['has_val'] == True:
            self.true_ui['val'] = data.user_items['val']
        self.true_ui['test'] = data.user_items['test']
        print(f"Basic_test got ready!")
    
    
    def run(self, model, istest=False, group_k=0):
        model.eval()
        #-----------------test or val------------------
        if istest == False and CFG['has_val']:
            true_ui = self.true_ui['val']
        else:
            true_ui = self.true_ui['test']

        if group_k > 1:
            all_result = dict()
            user_group = utils.user_group_split(true_ui, self.pos_ui, group_k)
            for key, all_user in user_group.items():
                result = epoch_test(model, self.pos_ui, true_ui, self.args, all_user)
                all_result[f"inter<{key}-{len(all_user)}"] = result
        else:
            all_result = epoch_test(model, self.pos_ui, true_ui, self.args)

        return all_result
