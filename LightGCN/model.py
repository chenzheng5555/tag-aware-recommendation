import torch
from torch import nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class LightGCN(MessagePassing):
    def __init__(self, data, config):
        super(LightGCN, self).__init__(aggr="add")
        self.num_users = data.num_users
        self.num_items = data.num_items
        self.latent_dim = config.latent_dim
        self.num_layers = config.layers
        #self.drop_rate = config.drop_rate

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users,
                                                 embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items,
                                                 embedding_dim=self.latent_dim)
        # init embedding
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.acf = nn.Sigmoid()

        self.edge_index = data.getEdges()
        self.norm = self.get_norm()
        print(f"LightGCN got ready!")

    def get_norm(self):
        source, target = self.edge_index
        deg = degree(target, num_nodes=self.num_users + self.num_items)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
        norm = deg_inv_sqrt[source] * deg_inv_sqrt[target]
        return norm

    def forward(self):
        all_emb = torch.cat(
            [self.embedding_user.weight, self.embedding_item.weight])
        embs = [all_emb]
        for _ in range(self.num_layers):
            all_emb = self.propagate(self.edge_index,x=all_emb,norm=self.norm)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def getUsersRating(self, users):
        all_users, all_items = self.forward()
        users_emb = all_users[users]
        items_emb = all_items
        rating = self.acf(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getBPRLoss(self, users, pos_items, neg_items):
        all_users, all_items = self.forward()
        users_emb = all_users[users]
        users_emb0 = self.embedding_user(users)
        pos_emb = all_items[pos_items]
        pos_emb0 = self.embedding_item(pos_items)
        neg_emb = all_items[neg_items]
        neg_emb0 = self.embedding_item(neg_items)

        reg_loss = (1 / 2) * (users_emb0.norm(2).pow(2) + \
                    pos_emb0.norm(2).pow(2) + neg_emb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        # 计算每个用户正样本与负样本的匹配差异
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        return loss, reg_loss