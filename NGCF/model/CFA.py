import torch
import torch.nn as nn


class CFA(nn.Module):
    def __init__(self, cfg):
        super(CFA, self).__init__()
        self.encoder = nn.Linear(cfg.layer_embed[0], cfg.layer_embed[1])
        self.decoder = nn.Linear(cfg.layer_embed[1], cfg.layer_embed[2])

    def forward(self, users):
        h = nn.functional.sigmoid(self.encoder(users))
        return h

    def bpr_loss(self, users, num_user, h):
        h = nn.functional.sigmoid(self.decoder(h))
        loss = nn.functional.mse_loss(h, users)

        return loss, 0
