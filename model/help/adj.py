import scipy.sparse as sp
import numpy as np
import torch


#------------------------create adj-----------------------------------
def create_ui_adj(ui_adj):
    n = sum(ui_adj.shape)
    n_user = ui_adj.shape[0]
    ui_adj = ui_adj.tolil()

    adj = sp.lil_matrix((n, n), dtype=np.float32)
    adj[:n_user, n_user:] = ui_adj
    adj[n_user:, :n_user] = ui_adj.T
    adj = adj.tocsr()
    return adj


def create_uit_adj(ui_adj, ut_adj, it_adj):
    n_ui = sum(ui_adj.shape)
    n = n_ui + ut_adj.shape[1]
    n_user = ui_adj.shape[0]
    ui_adj = ui_adj.tolil()
    ut_adj = ut_adj.tolil()
    it_adj = it_adj.tolil()

    adj = sp.lil_matrix((n, n), dtype=np.float32)
    adj[:n_user, n_user:n_ui] = ui_adj
    adj[:n_user, n_ui:] = ut_adj
    adj[n_user:n_ui, :n_user] = ui_adj.T
    adj[n_user:n_ui, n_ui:] = it_adj
    adj[n_ui:, :n_user] = ut_adj.T
    adj[n_ui:, n_user:n_ui] = it_adj.T
    adj = adj.tocsr()
    return adj


def creat_adj(data, use_tag, norm_type, split_adj_k, device):
    if use_tag:
        adj = create_uit_adj(data.ui_adj, data.ut_adj, data.it_adj)
    else:
        adj = create_ui_adj(data.ui_adj)
    adj = adj.tocsr()
    norm_adj = get_norm_adj(adj, norm_type)
    norm_adj = split_sp2tensor(norm_adj, split_adj_k, device)
    return norm_adj


def create_ui_t_adj(ut_adj, it_adj):
    n_ui = ut_adj.shape[0] + it_adj.shape[0]
    n = n_ui + ut_adj.shape[1]
    n_user = ut_adj.shape[0]
    #ui_adj = data.ui_adj.tolil()
    ut_adj = ut_adj.tolil()
    it_adj = it_adj.tolil()

    adj = sp.lil_matrix((n, n), dtype=np.float32)
    #adj[:n_user, n_user:n_ui] = ui_adj
    adj[:n_user, n_ui:] = ut_adj
    #adj[n_user:n_ui, :n_user] = ui_adj.T
    adj[n_user:n_ui, n_ui:] = it_adj
    adj[n_ui:, :n_user] = ut_adj.T
    adj[n_ui:, n_user:n_ui] = it_adj.T
    adj = adj.tocsr()
    return adj


def create_split_adj(row, col, val, n, k):
    adj = sp.coo_matrix((val, (row, col)), dtype=np.float32, shape=(n, n))
    adj_fold = split_sp_mat(adj, k)
    return adj_fold


#------------------------norm adj-----------------------------------
def get_norm_adj(adj, norm_type):
    if norm_type == "bi_norm":  # lightgcn
        norm_adj = bi_norm_laplacian(adj)
    elif norm_type == "si_norm":  # gcmc in ngcf paper
        norm_adj = si_norm_laplacian(adj)
    elif norm_type == "si_norm_self":  # norm in ngcf paper
        norm_adj = si_norm_laplacian(adj + sp.eye(adj.shape[0], dtype=adj.dtype))
    elif norm_type == "ngcf":  # default in ngcf paper
        norm_adj = si_norm_laplacian(adj) + sp.eye(adj.shape[0], dtype=adj.dtype)
    else:
        return adj

    return norm_adj


def bi_norm_laplacian(adj):
    adj = adj.tocsr()
    row_sum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()


def si_norm_laplacian(adj):
    adj = adj.tocsr()
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()


#------------------------split adj-----------------------------------
def split_sp_mat(matrix, k):
    if k < 2:
        return matrix
    matrix = matrix.tocsr()
    matrix_fold = []
    num = matrix.shape[0]
    fold_len = num // k
    for i in range(k):
        start = i * fold_len
        if i == k - 1:
            end = num
        else:
            end = start + fold_len
        adj = matrix[start:end]
        matrix_fold.append(adj)

    return matrix_fold


def split_sp2tensor(old_adj, k, device):
    if k > 1:
        new_adj = []
        for adj in split_sp_mat(old_adj, k):
            new_adj.append(sp2tensor(adj).to(device))
    else:
        new_adj = sp2tensor(old_adj).to(device)
    return new_adj


#------------------------convert adj-----------------------------------
def sp2tensor(norm_adj):
    norm_adj = norm_adj.tocoo()
    row = torch.from_numpy(norm_adj.row)
    col = torch.from_numpy(norm_adj.col)
    val = torch.from_numpy(norm_adj.data)
    adj = torch.sparse_coo_tensor(torch.stack([row, col]), val, norm_adj.shape)
    return adj


def tensor2sp(adj):
    raise NotImplementedError


#------------------------sparse adj @ tensor-----------------------------------
def split_mm(norm_adj, all_embed):
    nei_embed = []
    if isinstance(norm_adj, list):
        for adj in norm_adj:
            embed = torch.sparse.mm(adj, all_embed)
            nei_embed.append(embed)
        nei_embed = torch.cat(nei_embed, dim=0)
    else:
        nei_embed = torch.sparse.mm(norm_adj, all_embed)
    return nei_embed


def node_drop(graph, keep_prob, training=False):
    assert keep_prob >= 0 and keep_prob < 1
    if keep_prob == 0 or training == False:
        return graph

    def drop(x, k):
        index = x._indices().t()
        values = x._values()
        random_index = torch.rand(len(values)) + k
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / k
        g = torch.sparse_coo_tensor(index.t(), values, x.shape, device=x.device)
        return g

    graphs = []
    if isinstance(graph, list):
        for g in graph:
            graphs.append(drop(g, 1 - keep_prob))
    else:
        graphs = drop(graph, 1 - keep_prob)
    return graphs
