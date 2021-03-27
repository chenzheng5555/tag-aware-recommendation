import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from .utils import get_norm, get_all_rating, get_bpr_score


class LightGCN(MessagePassing):
    def __init__(self, cfg):
        super(LightGCN, self).__init__(aggr='add')
        self.drop_rate = cfg.drop_rate
        self.num_nodes = sum(cfg.node_list)
        self.num_layer = len(cfg.layer_embed) - 1
        self.embedding = nn.ModuleList()
        dim_k = cfg.layer_embed[0]
        for num_node in cfg.node_list:
            emb = nn.Embedding(num_node, dim_k)
            nn.init.normal_(emb.weight, std=0.1)
            self.embedding.append(emb)

    def forward(self, edge_index):
        # 添加自循环边
        edge_index, norm = get_norm(edge_index, self.num_nodes, False)
        x = torch.cat([i.weight for i in self.embedding])
        embed = [x]
        for _ in range(self.num_layer):
            x = self.propagate(edge_index, x=x, norm=norm)
            # x = F.dropout(x, p=self.drop_rate, training=self.training)
            embed.append(x)

        embed = torch.stack(embed, dim=1)
        light_out = torch.mean(embed, dim=1)

        return light_out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def loss(self, x, emb, cfg):
        init_emb = torch.cat([i.weight for i in self.embedding])

        pos_scores, neg_scores, reg_loss = get_bpr_score(x, emb, init_emb, cfg.node_list[0])
        loss = torch.mean(nn.functional.softplus(neg_scores - pos_scores))

        return loss + reg_loss * cfg.weight_decay

    def users_rating(self, users, emb, cfg):
        return get_all_rating(users, emb, cfg.node_list[0], cfg.node_list[1])
