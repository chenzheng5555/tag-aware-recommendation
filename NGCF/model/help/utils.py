import torch
import numpy as np
import scipy.sparse as sp


def np2tensor(np_data, device):
    new_data = None
    if isinstance(np_data, dict):
        new_data = dict()
        for u in np_data.keys():
            new_data[u] = torch.from_numpy(np_data[u]).to(device)

    elif isinstance(np_data, list):
        new_data = []
        for d in np_data:
            data = torch.from_numpy(d).to(device)
            new_data.append(data)

    else:
        new_data = torch.from_numpy(np_data).to(device)

    return new_data


def save_max(p_ui, dim=0):
    A = torch.zeros(p_ui.shape, dtype=p_ui.dtype, device=p_ui.device)
    r = torch.argmax(p_ui, dim=dim)
    c = torch.arange(len(r), device=p_ui.device)
    A[r, c] = 1
    return A