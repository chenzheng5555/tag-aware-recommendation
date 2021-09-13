from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F


class Layer(nn.Module):
    def __init__(self, fac_k, iter_k, in_dim, out_dim):
        super().__init__()
        self.fac_k = fac_k
        self.iter_k = iter_k
        self.in_dim = in_dim
        self.out_dim = out_dim
        self._init_weight()

    def _init_weight(self):
        dim_k = int(self.out_dim / self.fac_k)
        self.W = nn.Parameter(torch.Tensor(self.fac_k, self.in_dim, dim_k))
        self.b = nn.Parameter(torch.Tensor(self.fac_k, 1, dim_k))

    def forward(self, adj, all_emb):
        fac_emb = torch.matmul(all_emb, self.W + self.b)
        fac_emb = nn.LeakyReLU(negative_slope=0.2)(fac_emb)
        fac_emb = F.normalize(fac_emb, p=2, dim=2)
        row, col = adj._indices()
        new_fac_emb = fac_emb
        for t in range(self.iter_k):
            head = new_fac_emb[:, row.long()]
            tail = fac_emb[:, col.long()]
            p_uv = torch.sum(torch.mul(head, tail), dim=2)
            p_uv = torch.softmax(p_uv, dim=0)
            emb_list = []
            for i in range(self.fac_k):
                adj = torch.sparse_coo_tensor(adj._indices(), p_uv[i].detach().cpu(), \
                    adj.shape, device=all_emb.device)

                emb = torch.sparse.mm(adj, fac_emb[i])
                emb = fac_emb[i] + emb
                emb = F.normalize(emb, p=2, dim=1)
                emb_list.append(emb)
            new_fac_emb = torch.stack(emb_list)

        emb = torch.cat(list(new_fac_emb), dim=1)
        return emb


class DisenGCN(nn.Module):
    def __init__(self, data, args=None):
        super().__init__()
        self._config(CFG)
        self.num_list = [data.num['user'], data.num['item'], data.num['tag']]
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
        self.message_drop_list = config['message_drop_list']

    def _init_weight(self):
        self.embed = nn.ParameterList()
        for num in self.num_list:
            self.embed.append(nn.Parameter(torch.Tensor(num, self.dim_latent)))

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer.append(Layer(self.factor_k, self.iterate_k, self.dim_latent, self.dim_latent))

        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            initialer(p)

    def forward(self):
        all_emb = torch.cat(list(self.embed), dim=0)
        list_emb = [all_emb]
        for i in range(self.num_layer):
            all_emb = self.layer[i].forward(self.norm_adj, all_emb)
            all_emb = F.dropout(all_emb, p=self.message_drop_list[i], training=self.training)
            #norm_emb = F.normalize(all_emb, p=2, dim=1)
            #list_emb.append(norm_emb)

        #all_emb = torch.mean(torch.stack(list_emb, dim=1), dim=1)
        #all_emb = torch.cat(list_emb, dim=1)
        list_emb = torch.split(all_emb, self.num_list, dim=0)
        return list_emb

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
        # user_ego, item_ego = self.get_ego_embed()[:2]
        # users_emb = user_ego[users.long()]
        # pos_emb = item_ego[pos_items.long()]
        # neg_emb = item_ego[neg_items.long()]
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

        return loss, self.reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating