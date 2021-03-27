import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import get_bpr_score, get_all_rating


class MF(nn.Module):
    def __init__(self, cfg):
        super(MF, self).__init__()
        self.num_nodes = sum(cfg.node_list)
        self.embedding = nn.ModuleList()
        self.drop_rate = cfg.drop_rate
        dim_k = cfg.layer_embed[0]
        for num_node in cfg.node_list:
            emb = nn.Embedding(num_node, dim_k)
            nn.init.xavier_uniform_(emb.weight)
            self.embedding.append(emb)

    def forward(self, edge_index):
        # 添加自循环边
        x = torch.cat([i.weight for i in self.embedding])
        x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x

    def loss(self, data, emb, cfg):

        pos_scores, neg_scores, reg_loss = get_bpr_score(data, emb, emb, cfg.node_list[0])
        loss = - torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

        return loss + reg_loss * cfg.weight_decay

    def users_rating(self, users, emb, cfg):
        return get_all_rating(users, emb, cfg.node_list[0], cfg.node_list[1])
