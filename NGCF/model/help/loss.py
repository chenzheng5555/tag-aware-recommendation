import torch


def mul_loss(users_emb, pos_emb, neg_emb, loss_func):
    pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
    neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

    if loss_func == "logsigmoid":  # ngcf,tgcn
        loss = -1 * torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))
    else:
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss


# def fac_mul_loss(users_emb, pos_emb, neg_emb, dim_k):
#     fac_u_emb = torch.stacK(torch.split(users_emb, dim_k, dim=1))
#     fac_pi_emb = torch.stacK(torch.split(pos_emb, dim_k, dim=1))
#     fac_ni_emb = torch.stacK(torch.split(neg_emb, dim_k, dim=1))

#     att_p = torch.sum(torch.mul(fac_u_emb, fac_pi_emb), dim=2)


#     pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)
#     neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)


def l2reg_loss(*embs):
    loss = 0
    for emb in embs:
        loss += emb.norm(2).pow(2)
    reg_loss = (1 / 2) * (loss) / float(embs[0].shape[0])
    return reg_loss


def transtag_loss(head_e, rela_e, pos_tail_e, neg_tail_e, margin=0):
    pos_score = torch.norm(head_e + rela_e - pos_tail_e, p=2, dim=1)
    neg_score = torch.norm(head_e + rela_e - neg_tail_e, p=2, dim=1)

    kg_loss = torch.mean(torch.relu(margin + pos_score - neg_score))

    return kg_loss


def transe_loss(head_e, rela_e, pos_tail_e, neg_tail_e):
    pos_score = torch.norm(head_e + rela_e - pos_tail_e, p=2, dim=1)
    neg_score = torch.norm(head_e + rela_e - neg_tail_e, p=2, dim=1)

    kg_loss = torch.mean(torch.nn.functional.softplus(pos_score - neg_score))

    return kg_loss


def cor_loss(factor_emb, factor_k):
    def creat_centered_distance(X):
        r = torch.sum(torch.square(X), dim=1, keepdim=True)
        d = torch.sqrt(torch.maximum(r - 2 * torch.matmul(X, X.t()) + r.t(), torch.tensor(0.0)) + 1e-8)
        d = d - torch.mean(d, dim=0, keepdim=True) - torch.mean(d, dim=1, keepdim=True) + torch.mean(d)
        return d

    def create_distance_covariance(D1, D2):
        n_samples = D1.shape[0]
        dcov = torch.sqrt(torch.maximum(torch.sum(D1 * D2) / (n_samples * n_samples), torch.tensor(0.0)) + 1e-8)
        return dcov

    def create_distance_correlation(x, y):
        d1 = creat_centered_distance(x)
        d2 = creat_centered_distance(y)
        dcon_xy = create_distance_covariance(d1, d2)
        dcon_xx = create_distance_covariance(d1, d1)
        dcon_yy = create_distance_covariance(d2, d2)
        dcor = dcon_xy / (torch.sqrt(torch.maximum(dcon_xx * dcon_yy, torch.tensor(0.0))) + 1e-10)
        return dcor

    loss = 0
    for i in range(0, factor_k - 1):
        x = factor_emb[i]
        y = factor_emb[i + 1]
        loss += create_distance_correlation(x, y)

    loss /= ((factor_k + 1.0) * factor_k / 2)
    return loss