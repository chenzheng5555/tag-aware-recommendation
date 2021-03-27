import torch
import numpy as np
from sklearn.metrics import roc_auc_score


def NDCGatK_r(test_data, r, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg / idcg
    ndcg[np.isnan(ndcg)] = 0.
    return ndcg


def AUC(all_item_scores, num_item, test_data):
    """
    design for a single user
    """
    r_all = np.zeros((num_item, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)


def batch_metric(met, list_k):
    '''
    met : predict_l, groundtrue_k, ndcg, aucs
    '''
    k = len(list_k)
    predict_l = met[0:k]
    groundtrue_k = met[k]
    ndcg = met[k + 1:2 * k + 1]
    aucs = met[-1]
    pre = predict_l.mean(axis=1) / np.array(list_k)
    rec = (predict_l / groundtrue_k).mean(axis=1)
    ndc = ndcg.mean(axis=1)
    auc = aucs.mean()
    hr = predict_l.sum(axis=1) / groundtrue_k.sum()
    return [pre, rec, hr, ndc, auc]


def get_metrics(users, data, rating, list_k, all_user=True):
    '''
    all_user: True, compute the metrics in all users; False, in batch users
    '''
    pred = []
    groundtrue = []

    for i, u in enumerate(users):
        if u.item() in data.train_set.keys():
            train_pos = data.train_set[u]
            rating[i, train_pos] = -(1 << 10)

        test_pos = data.test_set[u]
        groundtrue.append(test_pos)
        _, rating_K = torch.topk(rating[i], k=max(list_k))
        pre = list(map(lambda x: x in test_pos, rating_K.numpy()))  # 如果rating_K为tensor,test_pos为array,则pre全为False
        pred.append(pre)

    pred = np.array(pred)
    predict_l = []
    ndcg = []
    for k in list_k:
        prec = pred[:, :k].sum(axis=1)
        predict_l.append(prec)
        ndcg.append(NDCGatK_r(groundtrue, pred, k))
    predict_l = np.array(predict_l)
    ndcg = np.array(ndcg)
    aucs = np.array([AUC(rating[i].numpy(), data.num_item, test_data) for i, test_data in enumerate(groundtrue)])
    groundtrue_k = np.array([len(i) for i in groundtrue])
    met = np.vstack((predict_l, groundtrue_k, ndcg, aucs))
    if all_user:
        return met
    else:
        return batch_metric(met, list_k)


def compute_all_metric(metrics, list_k, all_suer=True):
    dict_metric = {"precision": [], "recall": [], "ndcg": [], "HR": [], "auc": []}
    if all_suer:
        met = np.hstack(metrics)
        pre, rec, hr, ndc, auc=batch_metric(met,list_k)
    else:
        met = np.array(metrics,dtype=object)
        m =[met[:,i] for i in range(5)]
        pre, rec, hr, ndc, auc=[i.mean(axis=0) for i in m]

    dict_metric["precision"]=pre
    dict_metric["recall"]=rec
    dict_metric["ndcg"]=ndc
    dict_metric["HR"]=hr
    dict_metric["auc"]=[auc]

    for key, val in dict_metric.items():
        dict_metric[key] = list(map(lambda x: round(x, 5), val))

    return dict_metric
