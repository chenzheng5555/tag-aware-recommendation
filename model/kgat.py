from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class KGAT(nn.Module):
    def __init__(self, data, args=None):
        super().__init__()
        self._config(CFG)

        self.num_user = data.num['user']
        self.num_entity = data.num['item'] + data.num['tag']
        self.num_relation = 6
        # edge index
        edge_index_dict = data.create_edge()
        self.edge_index_dict = utils.np2tensor(edge_index_dict, self.device)

        self._init_weight()

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.dim_relation = config['dim_relation']
        self.dim_layer_list = config['dim_layer_list']
        self.num_layer = len(self.dim_layer_list)
        self.dim_layer_list = [self.dim_latent] + self.dim_layer_list
        self.agg_type = config['agg_type']
        self.device = config['device']
        self.message_drop_list = config['message_drop_list']
        self.split_adj_k = config["split_adj_k"]
        self.reg = config['reg']
        self.cor_reg = config['cor_reg']
        self.loss_func = config['mul_loss_func']

    def _init_weight(self):
        self.embed = nn.ParameterDict({
            "user": nn.Parameter(torch.Tensor(self.num_user, self.dim_latent)),
            "entity": nn.Parameter(torch.Tensor(self.num_entity, self.dim_latent)),
            "relation": nn.Parameter(torch.Tensor(self.num_relation, self.dim_relation)),
        })
        self.mat = nn.ParameterDict({
            "transE": nn.Parameter(torch.Tensor(self.num_relation, self.dim_latent, self.dim_relation)),
        })
        for k in range(self.num_layer):
            self.mat.update({
                f"W1_{k}": nn.Parameter(torch.Tensor(self.dim_layer_list[k], self.dim_layer_list[k + 1])),
                f"b1_{k}": nn.Parameter(torch.Tensor(1, self.dim_layer_list[k + 1])),
            })

            if self.agg_type == "bi_inter":
                self.mat.update({
                    f"W2_{k}": nn.Parameter(torch.Tensor(self.dim_layer_list[k], self.dim_layer_list[k + 1])),
                    f"b2_{k}": nn.Parameter(torch.Tensor(1, self.dim_layer_list[k + 1])),
                })

        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            initialer(p)

    def forward(self):
        all_embed = torch.cat([self.embed['user'], self.embed['entity']], dim=0)

        # update attention
        pai_list, row_list, col_list = [], [], []
        for k in self.edge_index_dict.keys():
            e_rela_mat = self.mat['transE'][k]
            e_rela = self.embed['relation'][k]
            row = self.edge_index_dict[k][:, 0].long()
            col = self.edge_index_dict[k][:, 1].long()
            e_head = all_embed[row]
            e_tail = all_embed[col]
            tr = torch.matmul(e_tail, e_rela_mat)
            hr = torch.matmul(e_head, e_rela_mat) + e_rela
            pai = torch.sum(tr * torch.tanh(hr), dim=1)
            pai_list.append(pai)
            row_list.append(row)
            col_list.append(col)

        # construct adj matrix
        val = torch.cat(pai_list)
        row = torch.cat(row_list)
        col = torch.cat(col_list)
        n = all_embed.shape[0]

        if self.split_adj_k > 1:
            norm_adj_fold = utils.create_split_adj(row.cpu().numpy(), col.cpu().numpy(), val.cpu().detach().numpy(), n, self.split_adj_k)
            norm_adj = []
            for adj in norm_adj_fold:
                adj = utils.sp2tensor(adj).to(self.device)
                adj = torch.sparse.softmax(adj, dim=1)
                norm_adj.append(adj)
        else:
            adj = torch.sparse.FloatTensor(torch.stack([row, col]), val, (n, n)).to(self.device)
            norm_adj = torch.sparse.softmax(adj, dim=1)
            # norm_adj = norm_adj.to_dense()

        if self.agg_type == "bi_inter":
            all_embed = self.bi_inter_embed(norm_adj, all_embed)

        user_embed, entity_embed = torch.split(all_embed, [self.num_user, self.num_entity], dim=0)
        return user_embed, entity_embed

    def bi_inter_embed(self, norm_adj, all_embed):
        all_embed_list = [all_embed]
        for k in range(self.num_layer):

            nei_embed = utils.split_mm(norm_adj, all_embed)

            sum_embed = nei_embed + all_embed
            sum_embed = torch.matmul(sum_embed, self.mat[f'W1_{k}'] + self.mat[f'b1_{k}'])
            sum_embed = nn.LeakyReLU(negative_slope=0.2)(sum_embed)
            bi_embed = nei_embed * all_embed
            bi_embed = torch.matmul(bi_embed, self.mat[f'W2_{k}'] + self.mat[f'b2_{k}'])
            bi_embed = nn.LeakyReLU(negative_slope=0.2)(bi_embed)
            all_embed = sum_embed + bi_embed
            all_embed = F.dropout(all_embed, p=self.message_drop_list[k], training=self.training)

            norm_embed = F.normalize(all_embed, p=2, dim=1)
            all_embed_list += [norm_embed]

        all_embed = torch.cat(all_embed_list, dim=1)
        return all_embed

    def get_embed(self, batch_data):
        head, rela, pos_tail, neg_tail = batch_data.T
        all_embed = torch.cat([self.embed['user'], self.embed['entity']], dim=0)

        r_e = self.embed['relation'][rela.long()]
        h_emb = all_embed[head.long()].unsqueeze(1)
        pos_t_emb = all_embed[pos_tail.long()].unsqueeze(1)
        neg_t_emb = all_embed[neg_tail.long()].unsqueeze(1)

        trans_E = self.mat['transE'][rela.long()]
        h_e = torch.matmul(h_emb, trans_E).squeeze()
        pos_t_e = torch.matmul(pos_t_emb, trans_E).squeeze()
        neg_t_e = torch.matmul(neg_t_emb, trans_E).squeeze()

        return h_e, r_e, pos_t_e, neg_t_e

    def loss(self, batch_data):
        users, pos_items, neg_items = batch_data.T
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)

        return loss, self.reg * reg_loss

    def transe_loss(self, batch_data):
        h_e, r_e, pos_t_e, neg_t_e = self.get_embed(batch_data)
        pos_score = torch.norm(h_e + r_e - pos_t_e, p=2, dim=1).pow(2)
        neg_score = torch.norm(h_e + r_e - neg_t_e, p=2, dim=1).pow(2)

        kg_loss = torch.mean(torch.nn.functional.softplus(pos_score - neg_score))
        reg_loss = utils.l2reg_loss(h_e, r_e, pos_t_e, neg_t_e)

        return kg_loss, self.cor_reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating