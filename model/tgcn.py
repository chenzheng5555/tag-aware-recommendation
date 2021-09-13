from utility.word import CFG
import model.help as utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


class Attention1(nn.Module):
    def __init__(self, in_features: int, atten_dim: int, dim_w: int):
        super().__init__()
        self.W_1 = nn.Parameter(torch.Tensor(in_features + dim_w, atten_dim))
        self.W_2 = nn.Parameter(torch.Tensor(in_features, atten_dim))
        self.b = nn.Parameter(torch.Tensor(1, atten_dim))
        self.v = nn.Parameter(torch.Tensor(1, atten_dim))
        self.act = nn.ReLU()

    def forward(self, ev, ej, ew, v_jw):
        zeroj = torch.zeros((1, ej.shape[1]), dtype=ej.dtype, device=ej.device)
        zerow = torch.zeros((1, ew.shape[1]), dtype=ew.dtype, device=ew.device)
        ej = torch.cat([zeroj, ej])
        ew = torch.cat([zerow, ew])
        v_j, v_w = v_jw
        k = v_j.shape[1]

        eNj = ej[v_j]
        eNw = ew[v_w]
        eNv = ev.unsqueeze(1).repeat(1, k, 1)

        eN_vw = torch.cat([eNv, eNw], dim=-1)
        av_j = torch.matmul(eN_vw, self.W_1) + torch.matmul(eNj, self.W_2) + self.b
        x = torch.matmul(self.act(av_j), self.v.T)
        a = torch.softmax(x, dim=1)
        eN = torch.sum(a * eNj, dim=1)
        return eN


class BasicLayer(nn.Module):
    def __init__(self, in_features, out_features, atten_dim, weight_dim, \
        num_bit_conv, num_vector_conv):
        super().__init__()
        self._in_dim = in_features
        self._num_vector_conv = num_vector_conv
        self._num_bit_conv = num_bit_conv

        self.atten1 = nn.ModuleDict()
        type_name = ["user", "item", "tag"]
        for i in range(3):
            self.atten1.update({f"{type_name[i]}": Attention1(in_features, atten_dim, weight_dim)})

        # attention2
        self.U = nn.Parameter(torch.Tensor(in_features, atten_dim))
        self.q = nn.Parameter(torch.Tensor(1, atten_dim))
        self.p = nn.Parameter(torch.Tensor(1, atten_dim))

        self.conv = self._conv_layer()
        # fusion
        in_k = self._num_bit_conv * in_features + self._num_vector_conv * (3 + 2 + 1)

        self.Wf = nn.Parameter(torch.Tensor(in_k, out_features))
        self.bf = nn.Parameter(torch.Tensor(1, out_features))
        self.act = nn.ReLU()

    def _conv_layer(self):
        vector_dict = nn.ModuleDict()
        for j in range(1, 4):
            vector_dict.update({
                f"conv_{j}": nn.Conv2d(1, self._num_vector_conv, kernel_size=(j, self._in_dim), bias=False),
            })
        conv_dict = nn.ModuleDict({
            "bit_level": nn.Conv2d(1, self._num_bit_conv, kernel_size=(3, 1), bias=False),
            "vec_level": vector_dict,
        })
        return conv_dict

    def _atten2(self, u, i, t):
        uit = torch.stack([u, i, t], dim=1)
        x = torch.matmul(uit, self.U) + self.q
        x = torch.matmul(self.act(x), self.p.T)
        b = torch.softmax(x, dim=1)
        x = b * uit
        return x

    def _conv(self, eN):
        x = eN.unsqueeze(dim=1)
        bit_e = self.conv["bit_level"](x)
        bit_e = self.act(bit_e)
        bit_e = bit_e.squeeze().reshape(bit_e.shape[0], -1)
        vec_e = []

        for model in self.conv["vec_level"].values():
            y = model(x)
            y = self.act(y)
            y = y.squeeze(dim=-1)
            vec_e.append(y.reshape(y.shape[0], -1))
        vector_e = torch.cat(vec_e, dim=-1)

        y = torch.cat([bit_e, vector_e], dim=1)
        return y

    def _fusion(self, x):
        x = torch.matmul(x, self.Wf) + self.bf
        x = self.act(x)
        return x

    def forward(self, eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw):
        eu_uN = eu
        eu_iN = self.atten1["item"].forward(eu, ei, ew, u_iw)
        eu_tN = self.atten1["tag"].forward(eu, et, ew, u_tw)
        ei_iN = ei
        ei_uN = self.atten1["user"].forward(ei, eu, ew, i_uw)
        ei_tN = self.atten1["tag"].forward(ei, et, ew, i_tw)
        et_tN = et
        et_uN = self.atten1["user"].forward(et, eu, ew, t_uw)
        et_iN = self.atten1["item"].forward(et, ei, ew, t_iw)

        euN = self._atten2(eu_uN, eu_iN, eu_tN)
        eiN = self._atten2(ei_uN, ei_iN, ei_tN)
        etN = self._atten2(et_uN, et_iN, et_tN)
        eu_c = self._conv(euN)
        ei_c = self._conv(eiN)
        et_c = self._conv(etN)
        eu_k = self._fusion(eu_c)
        ei_k = self._fusion(ei_c)
        et_k = self._fusion(et_c)

        # euuu = torch.cat([eu_uN, ei_uN, et_uN], 0)
        # eiii = torch.cat([eu_iN, ei_iN, et_iN], 0)
        # ettt = torch.cat([eu_tN, ei_tN, et_tN], 0)
        # eNNN = self._atten2(euuu, eiii, ettt)
        # eccc = self._conv(eNNN)
        # eooo = self._fusion(eccc)
        # eu_k, ei_k, et_k = torch.split(eooo, [eu.shape[0], ei.shape[0], et.shape[0]], dim=0)

        return eu_k, ei_k, et_k


class TGCN(nn.Module):
    def __init__(self, data):
        super().__init__()
        self._config(CFG)
        self.num_user = data.num['user']
        self.num_item = data.num['item']
        self.num_tag = data.num['tag']
        self.num_weight = data.num['weight']

        self._init_weight()

        self.data = data
        start = time.time()
        self.all_sample = data.get_all_neighbor()
        #data.get_sample_neighbor(self.neighbor_k)
        print(f"TGCN got ready! [neighbor sample time {time.time()-start}]")

    def _config(self, config):
        self.dim_latent = config['dim_latent']
        self.dim_weight = config['dim_weight']
        self.num_layer = len(config['dim_layer_list'])
        self.dim_layer_list = [self.dim_latent] + config['dim_layer_list']
        self.dim_atten = config['dim_atten']
        self.num_bit_conv = config['num_bit_conv']
        self.num_vec_conv = config['num_vec_conv']
        self.message_drop_list = config['message_drop_list']
        self.device = config['device']
        self.neighbor_k = config['neighbor_k']
        self.reg = config['reg']
        self.transtag_reg = config['transtag_reg']
        self.loss_func = config['mul_loss_func']
        self.margin = config['margin']

    def _init_weight(self):
        self.embed = nn.ParameterDict({
            "user": nn.Parameter(torch.Tensor(self.num_user, self.dim_latent)),
            "item": nn.Parameter(torch.Tensor(self.num_item, self.dim_latent)),
            "tag": nn.Parameter(torch.Tensor(self.num_tag, self.dim_latent)),
            "weight": nn.Parameter(torch.Tensor(self.num_weight, self.dim_weight)),
        })
        self.layer = nn.ModuleDict()
        for k in range(self.num_layer):
            self.layer.update({
                f'{k}':BasicLayer(self.dim_layer_list[k], self.dim_layer_list[k + 1],\
                    self.dim_atten, self.dim_weight, self.num_bit_conv, self.num_vec_conv),
            })

        initializer = nn.init.xavier_uniform_
        for param in self.parameters():
            if isinstance(param, nn.Parameter):
                initializer(param)
            elif isinstance(param, nn.Conv2d):
                initializer(param.weight)

    def sample(self):
        neighbor = []
        for adj_w in self.all_sample:
            indexs = np.arange(adj_w[0].shape[1])
            np.random.shuffle(indexs)
            nei = [x[:, :self.neighbor_k] for x in adj_w]
            nei = torch.tensor(nei, dtype=torch.long, device=self.device)
            neighbor.append(nei)
        return neighbor

    def forward(self):
        eu, ei = self.embed['user'], self.embed['item']
        et, ew = self.embed['tag'], self.embed['weight']
        embs_u, embs_i, embs_t = [eu], [ei], [et]
        #neighbor = []
        #for nei in self.data.get_sample_neighbor(self.neighbor_k):
        #    nei = torch.tensor(nei, dtype=torch.long, device=self.device)
        #    neighbor.append(nei)
        #u_iw, u_tw, i_uw, i_tw, t_uw, t_iw = neighbor
        for i, layer in enumerate(self.layer.values()):
            neighbor = self.sample()
            u_iw, u_tw, i_uw, i_tw, t_uw, t_iw = neighbor
            eu, ei, et = layer(eu, ei, et, ew, u_iw, u_tw, i_uw, i_tw, t_uw, t_iw)
            eu = F.dropout(eu, p=self.message_drop_list[i], training=self.training)
            ei = F.dropout(ei, p=self.message_drop_list[i], training=self.training)
            et = F.dropout(et, p=self.message_drop_list[i], training=self.training)
            eu_norm = F.normalize(eu, p=2, dim=1)
            ei_norm = F.normalize(ei, p=2, dim=1)
            et_norm = F.normalize(et, p=2, dim=1)
            embs_u.append(eu_norm)
            embs_i.append(ei_norm)
            embs_t.append(et_norm)

        embs_u = torch.cat(embs_u, dim=1)
        embs_i = torch.cat(embs_i, dim=1)
        embs_t = torch.cat(embs_t, dim=1)
        return embs_u, embs_i, embs_t

    def get_ego_embed(self):
        return self.embed['user'], self.embed['item'], self.embed['tag']

    def loss(self, batch_data):
        users, pos_items, neg_items = batch_data.T
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users.long()]
        pos_emb = all_items[pos_items.long()]
        neg_emb = all_items[neg_items.long()]

        loss = utils.mul_loss(users_emb, pos_emb, neg_emb, self.loss_func)
        # user_ego, item_ego = self.get_ego_embed()[:2]
        # users_emb = user_ego[users.long()]
        # pos_emb = item_ego[pos_items.long()]
        # neg_emb = item_ego[neg_items.long()]
        reg_loss = utils.l2reg_loss(users_emb, pos_emb, neg_emb)

        return loss, self.reg * reg_loss

    def transtag_loss(self, batch_data):
        user, tag, pos_item, neg_item = batch_data.T
        all_users, all_items, all_tags = self.get_ego_embed()  #self.forward()
        tag_emb = all_tags[tag.long()]
        user_emb = all_users[user.long()]
        pos_i_emb = all_items[pos_item.long()]
        neg_i_emb = all_items[neg_item.long()]

        loss = utils.transtag_loss(user_emb, tag_emb, pos_i_emb, neg_i_emb, self.margin)
        reg_loss = utils.l2reg_loss(user_emb, tag_emb, pos_i_emb, neg_i_emb)
        return loss, self.transtag_reg * reg_loss

    def predict_rating(self, users):
        all_users, all_items = self.forward()[:2]
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.nn.Sigmoid()(torch.matmul(users_emb, items_emb.t()))
        return rating
