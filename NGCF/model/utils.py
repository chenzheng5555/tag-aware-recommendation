import torch
from torch_geometric.utils import add_self_loops, degree


def get_norm(edge_index, num_nodes, add_self_loop=True, p=-0.5, norm=0):
    if add_self_loop:
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

    source, target = edge_index[0], edge_index[1]
    deg = degree(target, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow_(p)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    if norm == 0:
        weight = deg_inv_sqrt[source] * deg_inv_sqrt[target]
    elif norm == 1:
        weight = deg_inv_sqrt[target]
    # # node dorpout, time consuming
    # if drop_node != 0 and training: 
    #     for i in range(num_nodes):
    #         index = target==i
    #         weight[index] = torch.dropout(weight[index], p=drop_node, train=training)
    return edge_index, weight


def get_all_rating(users, emb, num_user, num_item):
    all_item = torch.arange(num_user, num_user + num_item)
    users_emb = emb[users.long()]
    items_emb = emb[all_item.long()]
    rating = torch.sigmoid(torch.matmul(users_emb, items_emb.t()))

    return rating


def get_bpr_score(x, emb, init_emb, num_user):
    users, pos, neg = x[0], x[1], x[2]
    pos += num_user
    neg += num_user
    users_emb = emb[users.long()]
    pos_emb = emb[pos.long()]
    neg_emb = emb[neg.long()]
    pos_scores = torch.sum(users_emb * pos_emb, dim=1)
    neg_scores = torch.sum(users_emb * neg_emb, dim=1)

    if init_emb != None:
        user_emb0 = init_emb[users.long()]
        pos_emb0 = init_emb[pos.long()]
        neg_emb0 = init_emb[neg.long()]
        reg_loss = (1 / 2) * (user_emb0.norm(2).pow(2) + pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2)) / float(len(users))

    return pos_scores, neg_scores, reg_loss
