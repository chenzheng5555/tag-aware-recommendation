import numpy as np
import time
from collections import defaultdict
from sklearn.metrics import roc_auc_score

#---------------------------------------------------------
def get_label(true_ui, rating):
    label_list = []
    for i in range(len(rating)):
        label = list(map(lambda x: x in true_ui[i], rating[i]))
        label_list.append(label)
    return np.array(label_list, dtype=np.float32)


def pre_rec_k(label, true_ui, k):
    right = label[:, :k].sum(1)
    pre = np.sum(right) / k
    true_r = np.array([len(true_ui[i]) for i in range(len(true_ui))])
    rec = np.sum(right / true_r)
    hit = np.sum(right > 0)
    return {'recall': rec, 'precision': pre, 'hr': hit}


def ndcg_k(label, true_ui, k):
    right = label[:, :k]
    g_true = np.zeros((len(true_ui), k))
    for i in range(len(true_ui)):
        length = k if k < len(true_ui[i]) else len(true_ui[i])
        g_true[i, :length] = 1
    idcg = np.sum(g_true / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = np.sum(right / np.log2(np.arange(2, k + 2)), axis=1)
    idcg[idcg == 0.] = 1
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def auc(all_item_scores, num_item, test_data):
    """
    design for a single user
    """
    r_all = np.zeros((num_item, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

#-----------------------预测--------------------------------------
def minibatch(data, batch_size):
    step = len(data) // batch_size + 1
    for i in range(0, step):
        if (i + 1) * batch_size > len(data):
            yield data[i * batch_size:]
        else:
            yield data[i * batch_size:(i + 1) * batch_size]  # i+batch_size超出，不会出错


#---------------------group user---------------------------------------
def user_group_split(test_ui, train_ui, k, method="interaction"):
    start = time.time()
    all_user = list(test_ui.keys())
    num_inter = defaultdict(list)
    tot_inter = 0
    for u in all_user:
        n_inter = len(test_ui[u])
        if u in train_ui.keys():
            n_inter += len(train_ui[u])
        num_inter[n_inter].append(u)
        tot_inter += n_inter

    def by_interactin(tot, f):
        split_list_user = dict()
        step = tot // k
        end = list(range(step, tot + 1, step))
        end[-1] = tot

        count = 0
        i = 0
        temp = []
        for n in sorted(num_inter):
            temp += num_inter[n]

            if f == 0:
                count += n * len(num_inter[n])
            elif f == 1:
                count += len(num_inter[n])
            elif f == 2:
                count = n
            else:
                count += 1

            if count >= end[i]:
                split_list_user[n] = temp
                temp = []
                i += 1
                print(f"interaction < {n} has {len(split_list_user[n])} user")

        return split_list_user

    if method == "interaction":  # ngcf
        split_group_user = by_interactin(tot_inter, 0)
    elif method == "user":  #[0, user]
        split_group_user = by_interactin(len(all_user), 1)
    elif method == "interval":  #[0, max(n)]
        split_group_user = by_interactin(max(num_inter.keys()), 2)
    else:  #[0, len(n)]
        split_group_user = by_interactin(len(num_inter.keys()), 3)

    print(f"group user into {k} group, time{(time.time()-start)/60:.2}")
    return split_group_user
