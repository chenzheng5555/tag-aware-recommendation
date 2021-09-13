from torch.functional import norm
import model.help as utils
from utility.word import CFG

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightGCN(nn.Module):
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
        self.reg = config['reg']
        self.loss_func = config['mul_loss_func']
        self.use_tag = config['use_tag']
        self.message_drop_list = config['message_drop_list']
        self.node_drop = config['node_drop']

    def _init_weight(self):
        # embedding
        self.embed = nn.ParameterList()
        for num in self.num_list:
            self.embed.append(nn.Parameter(torch.Tensor(num, self.dim_latent)))

        # init
        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            #nn.init.normal_(p, std=0.1)
            initialer(p)

    def forward(self):
        norm_adj = utils.node_drop(self.norm_adj, self.node_drop, self.training)

        all_embed = torch.cat(list(self.embed), dim=0)
        all_embed_list = [all_embed]
        for k in range(self.num_layer):
            all_embed = utils.split_mm(norm_adj, all_embed)
            all_embed = F.dropout(all_embed, p=self.message_drop_list[k], training=self.training)
            norm_embed = F.normalize(all_embed, p=2, dim=1)  # 添加norm降低性能
            all_embed_list += [norm_embed]

        all_embed = torch.mean(torch.stack(all_embed_list, dim=1), dim=1)
        #all_embed = torch.cat(all_embed_list, dim=1)
        list_embed = torch.split(all_embed, self.num_list, dim=0)
        return list_embed

    def get_ego_embed(self):
        return list(self.embed)

    def loss(self, batch_data):
        users, pos_items, neg_items = batch_data.T
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        user_ego, item_ego = self.get_ego_embed()[:2]
        users_emb = user_ego[users.long()]
        pos_emb = item_ego[pos_items.long()]
        neg_emb = item_ego[neg_items.long()]
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)

        return loss, self.reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating
