import model.help as utils
from utility.word import CFG

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGCF(nn.Module):
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

        print(f"NGCF got ready!!!")

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.dim_layer_list = config['dim_layer_list']
        self.num_layer = len(self.dim_layer_list)
        self.dim_layer_list = [self.dim_latent] + self.dim_layer_list
        self.agg_type = config['agg_type']
        self.device = config['device']
        self.message_drop_list = config['message_drop_list']
        self.norm_type = config['norm_type']
        self.split_adj_k = config["split_adj_k"]
        self.reg = config['reg']
        self.loss_func = config['mul_loss_func']
        self.use_tag = config['use_tag']

    def _init_weight(self):
        # embedding
        self.embed = nn.ParameterList()
        for num in self.num_list:
            self.embed.append(nn.Parameter(torch.Tensor(num, self.dim_latent)))
        # matrix
        self.mat = nn.ParameterDict()
        for k in range(self.num_layer):
            self.mat.update({
                f"W1_{k}": nn.Parameter(torch.Tensor(self.dim_layer_list[k], self.dim_layer_list[k + 1])),
                f"b1_{k}": nn.Parameter(torch.Tensor(1, self.dim_layer_list[k + 1])),
            })

            if self.agg_type == "bi_agg":
                self.mat.update({
                    f"W2_{k}": nn.Parameter(torch.Tensor(self.dim_layer_list[k], self.dim_layer_list[k + 1])),
                    f"b2_{k}": nn.Parameter(torch.Tensor(1, self.dim_layer_list[k + 1])),
                })
        # init
        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            initialer(p)

    def forward(self):
        all_embed = torch.cat(list(self.embed), dim=0)

        if self.agg_type == "bi_agg":
            all_embed = self.bi_inter_embed(all_embed)
        else:
            raise NotImplementedError

        list_embed = torch.split(all_embed, self.num_list, dim=0)
        return list_embed

    def bi_inter_embed(self, all_embed):
        all_embed_list = [all_embed]
        for k in range(self.num_layer):
            nei_embed = utils.split_mm(self.norm_adj, all_embed)
            sum_embed = nei_embed + all_embed
            sum_embed = torch.matmul(sum_embed, self.mat[f'W1_{k}'] + self.mat[f'b1_{k}'])
            sum_embed = nn.LeakyReLU(negative_slope=0.2)(sum_embed)

            bi_embed = torch.mul(nei_embed, all_embed)
            bi_embed = torch.matmul(bi_embed, self.mat[f'W2_{k}'] + self.mat[f'b2_{k}'])
            bi_embed = nn.LeakyReLU(negative_slope=0.2)(bi_embed)
            all_embed = sum_embed + bi_embed
            all_embed = F.dropout(all_embed, p=self.message_drop_list[k], training=self.training)
            norm_embed = F.normalize(all_embed, p=2, dim=1)
            all_embed_list += [norm_embed]

        all_embed = torch.cat(all_embed_list, dim=1)
        return all_embed

    def get_ego_emb(self):
        return list(self.embed)

    def loss(self, batch_data):
        users, pos_items, neg_items = batch_data.T
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)

        return loss, self.reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating
