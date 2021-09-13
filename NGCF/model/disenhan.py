from numpy.core.fromnumeric import shape
from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np


class Layer(nn.Module):
    def __init__(self, factor_k, dim_in, dim_out, rela_k, iterate):
        super().__init__()
        self.factor_k = factor_k
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.rela_k = rela_k
        self.iterate = iterate
        self._init_weight()

    def _init_weight(self):
        dim_k = int(self.dim_out / self.factor_k)
        self.Wtk = nn.Parameter(torch.Tensor(3, self.factor_k, self.dim_in, dim_k))
        self.at = nn.Parameter(torch.Tensor(3 * 2, self.factor_k, 2 * dim_k))
        self.W = nn.Parameter(torch.Tensor(dim_k, dim_k))
        self.q_rela = nn.Parameter(torch.Tensor(3 * 2, dim_k))

    def forward(self, edge_list, shape_list, u_emb, i_emb, t_emb):
        def fac(emb, W):
            fac_emb = torch.matmul(emb, W)
            fac_emb = nn.LeakyReLU(negative_slope=0.2)(fac_emb)
            fac_emb = F.normalize(fac_emb, p=2, dim=2)
            return fac_emb

        def rela_update(indices, new_emb, old_emb, a, r_node, shape, q_rela):
            u, i = indices
            all_u = new_emb[:, u.long()]
            all_i = old_emb[:, i.long()]
            ui = torch.cat([all_u, all_i], dim=2)  # 4, 18614, 32
            e_ts = torch.matmul(ui, a.unsqueeze(2)).squeeze()
            e_ts_k = torch.relu(e_ts)

            r_k_edge = r_node[:, u.long()]
            e_ts_rela = torch.mul(e_ts_k, r_k_edge)
            e_ts_rela = torch.sum(e_ts_rela, dim=0)  #edges

            adj = torch.sparse_coo_tensor(indices, e_ts_rela, shape, device=old_emb.device)
            adj = torch.sparse.softmax(adj, dim=1)
            emb_z = []

            for k in range(self.factor_k):
                zk = torch.sparse.mm(adj, old_emb[k])
                zk = nn.LeakyReLU(negative_slope=0.2)(zk)
                emb_z.append(zk)
            emb_z = torch.stack(emb_z)
            emb_z = torch.matmul(emb_z, self.W)
            new_r = torch.matmul(torch.tanh(emb_z), q_rela)
            r = torch.softmax(new_r, dim=0)
            return r, emb_z

        def new_fac(ego_emb, r_list, e_list, i_list):
            for i in i_list:
                ego_emb = ego_emb + torch.mul(e_list[i], r_list[i].unsqueeze(2))
            ego_emb = F.normalize(ego_emb, p=2, dim=2)
            return ego_emb

        # 4*Nx*16
        fac_u_emb = fac(u_emb, self.Wtk[0])
        fac_i_emb = fac(i_emb, self.Wtk[1])
        fac_t_emb = fac(t_emb, self.Wtk[2])

        r_rela_list = []
        for e in range(len(shape_list)):
            r_rela_list.append(torch.ones((self.factor_k, shape_list[e][0])) / self.factor_k)

        all_ego_emb = [fac_u_emb, fac_i_emb, fac_t_emb]
        all_new_emb = all_ego_emb
        index = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]]
        for t in range(self.iterate):
            rela_list = []
            emb_list = []
            for e in range(len(edge_list)):
                new_emb = all_new_emb[index[e][0]]
                ego_emb = all_ego_emb[index[e][1]]
                new_r, new_e = rela_update(edge_list[e], new_emb, ego_emb, \
                    self.at[e], r_rela_list[e], shape_list[e], self.q_rela[e])
                rela_list.append(new_r)
                emb_list.append(new_e)

            u_e = new_fac(all_ego_emb[0], rela_list, emb_list, [0, 2])
            i_e = new_fac(all_ego_emb[1], rela_list, emb_list, [1, 4])
            t_e = new_fac(all_ego_emb[2], rela_list, emb_list, [3, 5])

            all_new_emb = [u_e, i_e, t_e]
            r_rela_list = rela_list

        emb_list = []
        for e in all_new_emb:
            emb_list.append(torch.cat(list(e), dim=1))
        return emb_list


class DisenHAN(nn.Module):
    def __init__(self, data, args=None):
        super().__init__()
        self._config(CFG)
        self.num_list = [data.num['user'], data.num['item'], data.num['tag']]
        self.edge_list, self.shape_list = self.create_edge(data)
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

    def create_edge(self, data):
        edge_list, shape_list = [], []
        # ui iu ut tu it ti
        user, item = data.ui_adj.row, data.ui_adj.col
        edge_list.append(torch.from_numpy(np.stack([user, item])))
        edge_list.append(torch.from_numpy(np.stack([item, user])))
        shape_list.append((self.num_list[0], self.num_list[1]))
        shape_list.append((self.num_list[1], self.num_list[0]))

        user, tag = data.ut_adj.row, data.ut_adj.col
        edge_list.append(torch.from_numpy(np.stack([user, tag])))
        edge_list.append(torch.from_numpy(np.stack([tag, user])))
        shape_list.append((self.num_list[0], self.num_list[2]))
        shape_list.append((self.num_list[2], self.num_list[0]))

        item, tag = data.it_adj.row, data.it_adj.col
        edge_list.append(torch.from_numpy(np.stack([item, tag])))
        edge_list.append(torch.from_numpy(np.stack([tag, item])))
        shape_list.append((self.num_list[1], self.num_list[2]))
        shape_list.append((self.num_list[2], self.num_list[1]))

        return edge_list, shape_list

    def _init_weight(self):
        self.embed = nn.ParameterList()
        for num in self.num_list:
            self.embed.append(nn.Parameter(torch.Tensor(num, self.dim_latent)))

        self.layer = nn.ModuleList()
        for i in range(self.num_layer):
            self.layer.append(Layer(self.factor_k, self.dim_latent, self.dim_latent, 6, 2))

        initialer = nn.init.xavier_uniform_
        for p in self.parameters():
            initialer(p)

    def forward(self):
        user_emb, item_emb, tag_emb = list(self.embed)
        #u_layer_emb, i_layer_emb, t_layer_emb = [user_emb], [item_emb], [tag_emb]
        for i in range(self.num_layer):
            user_emb, item_emb, tag_emb = self.layer[i].forward(self.edge_list, self.shape_list, user_emb, item_emb, tag_emb)
            # user_emb = F.dropout(user_emb, p=self.message_drop_list[i], training=self.training)
            # item_emb = F.dropout(item_emb, p=self.message_drop_list[i], training=self.training)
            # tag_emb = F.dropout(tag_emb, p=self.message_drop_list[i], training=self.training)
            # norm_u_emb = F.normalize(user_emb, p=2, dim=1)
            # norm_i_emb = F.normalize(item_emb, p=2, dim=1)
            # norm_t_emb = F.normalize(tag_emb, p=2, dim=1)
            # u_layer_emb.append(norm_u_emb)
            # i_layer_emb.append(norm_i_emb)
            # t_layer_emb.append(norm_t_emb)

        # user_emb = torch.cat(u_layer_emb, dim=1)
        # item_emb = torch.cat(i_layer_emb, dim=1)
        # tag_emb = torch.cat(t_layer_emb, dim=1)

        return user_emb, item_emb, tag_emb

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