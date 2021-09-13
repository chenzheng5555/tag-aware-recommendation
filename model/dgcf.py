from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class DGCF(nn.Module):
    def __init__(self, data, args=None):
        super().__init__()
        self._config(CFG)

        if self.use_tag:
            self.num_list = [data.num['user'], data.num['item'], data.num['tag']]
        else:
            self.num_list = [data.num['user'], data.num['item']]
        self.norm_adj = utils.creat_adj(data, self.use_tag, self.norm_type, \
            self.split_adj_k, self.device)
        self._init_weight()

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.num_layer = len(config['dim_layer_list'])
        self.device = config['device']
        self.norm_type = config['norm_type']
        self.split_adj_k = config["split_adj_k"]
        self.factor_k = config['factor_k']
        self.iterate_k = config['iterate_k']
        self.dim_k = self.dim_latent // self.factor_k
        self.reg = config['reg']
        self.cor_reg = config['cor_reg']
        self.loss_func = config['mul_loss_func']
        self.use_tag = config['use_tag']

    def _init_weight(self):
        # embedding
        self.embed = nn.ParameterList()
        for num in self.num_list:
            self.embed.append(nn.Parameter(torch.Tensor(num, self.dim_latent)))

        # init
        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            initialer(p)

    def forward(self, out_A=False):
        A_values = torch.ones(self.factor_k, self.norm_adj._nnz(), device=self.device)
        ego_embed = torch.cat(list(self.embed), dim=0)
        all_embed = [ego_embed]
        out_layer_A = []
        for k in range(self.num_layer):
            A_values, ego_embed, out_A_factor = self.iterate_update(A_values, ego_embed)

            all_embed.append(ego_embed)
            out_layer_A.append(out_A_factor)

        all_embed = torch.stack(all_embed, dim=1)
        all_embed = torch.mean(all_embed, dim=1)
        list_emb = torch.split(all_embed, self.num_list, dim=0)
        if out_A == True:
            return out_layer_A

        return list_emb

    def iterate_update(self, A_values, ego_embed):
        ego_split_emb = torch.split(ego_embed, self.dim_k, dim=1)
        layer_emb = []
        out_A_factor = []
        for t in range(self.iterate_k):
            A_score_list = []
            A_factor = torch.softmax(A_values, dim=0)

            for i in range(self.factor_k):
                adj, factor_emb, A_score = self.factor_update(A_factor[i], ego_split_emb[i], self.norm_adj)
                A_score_list.append(A_score)
                if t == self.iterate_k - 1:
                    layer_emb.append(factor_emb)
                    out_A_factor.append(adj)

            A_score = torch.stack(A_score_list, dim=0)
            A_values += A_score

        layer_emb = torch.stack(layer_emb)
        layer_emb = F.normalize(layer_emb, p=2, dim=2)
        ego_embed = torch.cat(list(layer_emb), dim=1)
        return A_values, ego_embed, out_A_factor

    def factor_update(self, A_factor, ego_split_emb, norm_adj):
        adj = torch.sparse_coo_tensor(norm_adj._indices(), A_factor.detach().cpu(), \
            norm_adj.shape, device=self.device)
        col_sum = torch.sparse.sum(adj, dim=1)
        val = 1 / torch.sqrt(col_sum._values())
        val[torch.isinf(val)] = 0.0
        D = torch.sparse_coo_tensor(col_sum._indices()[0].unsqueeze(0).repeat(2, 1), val, \
            norm_adj.shape, device=self.device)
        factor_emb = torch.sparse.mm(D, ego_split_emb)
        factor_emb = torch.sparse.mm(adj, factor_emb)
        factor_emb = torch.sparse.mm(D, factor_emb)

        head, tail = norm_adj._indices().to(self.device)
        h_emb = factor_emb[head]
        t_emb = ego_split_emb[tail]  # t_emb = factor_emb[tail]
        h_emb = F.normalize(h_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)
        mut_emb = torch.mul(h_emb, torch.tanh(t_emb))
        A_score = torch.sum(mut_emb, dim=1)
        return adj, factor_emb, A_score

    def get_ego_embed(self):
        return list(self.embed)

    def loss(self, batch_data):
        data, cor = batch_data
        users, pos_items, neg_items = data.T
        all_embs = self.forward()
        all_users, all_items = all_embs[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        #---------------------reg-------------------------
        user_ego, item_ego = self.get_ego_embed()[:2]
        users_emb = user_ego[users.long()]
        pos_emb = item_ego[pos_items.long()]
        neg_emb = item_ego[neg_items.long()]
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)
        #-------------------cor-------------------------
        # emb_list = []
        # cor_user, cor_item = cor[:2]
        # emb_list.append(all_users[cor_user.long()])
        # emb_list.append(all_items[cor_item.long()])
        # if self.use_tag:
        #     emb_list.append(all_embs[2][cor[2].long()])

        # emb_list = all_embs
        # all_emb = torch.cat(emb_list, dim=0)
        # dim_k = int(all_emb.shape[1] / self.factor_k)
        # factor_emb = torch.split(all_emb, dim_k, dim=1)
        # cor_loss = utils.cor_loss(factor_emb, self.factor_k)

        return loss, self.reg * reg_loss#, self.cor_reg * cor_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating
