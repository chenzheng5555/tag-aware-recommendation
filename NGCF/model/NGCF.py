import torch
from torch.nn import Embedding, init, ModuleList, Linear
from torch_geometric.nn import MessagePassing
from .utils import get_norm, get_all_rating, get_bpr_score
import torch.nn.functional as F

# # class every GCN layer
# class GCN(MessagePassing):
#     def __init__(self, in_fea, out_fea, num_node, drop_rate):
#         super(GCN, self).__init__(aggr='add')
#         self.num_node = num_node
#         self.drop_rate = drop_rate
#         self.liner1 = Linear(in_fea, out_fea)
#         init.xavier_uniform_(self.liner1.weight)  # 似乎没有影响
#         self.liner2 = Linear(in_fea, out_fea)
#         init.xavier_uniform_(self.liner2.weight)

#     def forward(self, x, edge_index):
#         edge_index, norm = get_norm(edge_index, self.num_node, add_self_loop=True, p=-1, norm=1)
#         # # paper function
#         # out = self.propagate(edge_index, x=x, norm=norm)
#         # out += self.liner1(x)
#         # return F.leaky_relu(out)
#         emb = self.propagate(edge_index, x=x, norm=norm)
#         sum_emb = self.liner1(emb)
#         bi_emb = x * emb
#         bi_emb = self.liner2(bi_emb)
#         return F.leaky_relu(sum_emb + bi_emb, negative_slope=0.2)

#     def message(self, x_i, x_j, norm):
#         # # paper function
#         # return norm.view(-1, 1) * (self.liner1(x_j) + self.liner2(x_i * x_j))
#         drop = F.dropout(norm, p=0.9, training=self.training)
#         return drop.view(-1, 1) * x_j

# class NGCF(torch.nn.Module):
#     def __init__(self, cfg):
#         super(NGCF, self).__init__()
#         self.drop_rate = cfg.drop_rate
#         self.num_layer = len(cfg.layer_embed) - 1
#         self.num_node = sum(cfg.node_list)
#         self.gcn = ModuleList()
#         self.embedding = ModuleList()
#         for num_node in cfg.node_list:
#             emb = Embedding(num_node, cfg.layer_embed[0])
#             init.xavier_uniform_(emb.weight)  # 加上，性能有所提升
#             self.embedding.append(emb)
#         for i in range(self.num_layer):
#             self.gcn.append(GCN(cfg.layer_embed[i], cfg.layer_embed[i + 1], self.num_node, self.drop_rate))

#     def forward(self, edge_index):

#         x = torch.cat([i.weight for i in self.embedding])
#         out = [x]
#         for i in range(self.num_layer):
#             x = self.gcn[i](x, edge_index)
#             x = F.dropout(x, p=self.drop_rate, training=self.training)
#             x = F.normalize(x, p=2, dim=1)
#             out.append(x)

#         out = torch.cat(out, dim=1)
#         return out


class NGCF(MessagePassing):
    def __init__(self, cfg):
        super(NGCF, self).__init__(aggr='add')
        self.drop_rate = cfg.drop_rate
        self.drop_node = cfg.drop_node
        self.num_layer = len(cfg.layer_embed) - 1
        self.num_node = sum(cfg.node_list)
        self.embedding = ModuleList()
        self.lin1 = ModuleList()
        self.lin2 = ModuleList()
        for num_node in cfg.node_list:
            emb = Embedding(num_node, cfg.layer_embed[0])
            init.xavier_uniform_(emb.weight)
            self.embedding.append(emb)

        for i in range(self.num_layer):
            lin1 = Linear(cfg.layer_embed[i], cfg.layer_embed[i + 1])
            init.xavier_uniform_(lin1.weight)
            lin2 = Linear(cfg.layer_embed[i], cfg.layer_embed[i + 1])
            init.xavier_uniform_(lin2.weight)
            self.lin1.append(lin1)
            self.lin2.append(lin2)

    def forward(self, edge_index):
        x = torch.cat([i.weight for i in self.embedding])
        # edge_index, norm = get_norm(edge_index, self.num_node, add_self_loop=False)
        edge_index, norm = get_norm(edge_index, self.num_node, add_self_loop=True, p=-1.0, norm=1)
        norm = F.dropout(norm, p=self.drop_node, training=self.training)
        out = [x]
        for i in range(self.num_layer):
            # x_l = x
            # y = self.propagate(edge_index, x=x, norm=norm, lin1=self.lin1[i], lin2=self.lin2[i])
            # y += self.lin1[i](x_l)

            s = self.propagate(edge_index, x=x, norm=norm)
            b1 = self.lin1[i](s)
            b2 = self.lin2[i](s * x)
            y = b1 + b2

            x = F.leaky_relu(y, negative_slope=0.2)
            x = F.dropout(x, p=self.drop_rate, training=self.training)
            n = F.normalize(x, p=2, dim=1)
            out.append(n)

        out = torch.cat(out, dim=1)
        return out

    # def message(self, x_i, x_j, norm, lin1, lin2):
    #     return norm.view(-1, 1) * (lin1(x_j) + lin2(x_i * x_j))

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def loss(self, x, emb, cfg):
        init_emb = torch.cat([i.weight for i in self.embedding])

        pos_scores, neg_scores, reg_loss = get_bpr_score(x, emb, init_emb, cfg.node_list[0])
        loss = -1 * torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

        return loss + reg_loss * cfg.weight_decay

    def users_rating(self, users, emb, cfg):
        return get_all_rating(users, emb, cfg.node_list[0], cfg.node_list[1])