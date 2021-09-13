_ngcf = {
    "norm_type": "ngcf",
    "agg_type": "bi_agg",
    "mul_loss_func": "logsigmoid",
    #"message_drop_list": [0.5, 0.5, 0.5],
}

_lightgcn = {
    "mul_loss_func": "softplus",
    "norm_type": "bi_norm",
    'cor_batch': 100,
}

_dgcf = {
    "mul_loss_func": "softplus",
    "norm_type": "plain",
    'factor_k': 4,
    'iterate_k': 2,
    #'cor_reg': 1e-4,
    'cor_batch': 100,
}

_disengcn = {
    "mul_loss_func": "softplus",
    "norm_type": "plain",
    'factor_k': 4,
    'iterate_k': 2,
    #'cor_reg': 1e-4,
    'cor_batch': 100,
}

_disenhan = {
    "mul_loss_func": "softplus",
    "norm_type": "plain",
    'factor_k': 4,
    'iterate_k': 2,
    #'cor_reg': 1e-4,
    'cor_batch': 100,
}

_tgcn = {
    "dim_weight": 10,
    "dim_atten": 32,
    "num_bit_conv": 32,
    "num_vec_conv": 8,
    "margin": 1,
    "transtag_batch": 512,
    "neighbor_k": 25,
    'transtag_reg': 0.0001,
    "mul_loss_func": "logsigmoid",
    #"message_drop_list": [0.7, 0.7, 0.7],
}

_kgat = {
    'dim_relation': 64,
    'transe_reg': 0.0001,
    "transe_batch": 1024,
    "agg_type": "bi_agg",
    "mul_loss_func": "softplus",
}

_dtag = {
    "mul_loss_func": "softplus",
    "norm_type": "plain",
    'factor_k': 4,
    'iterate_k': 2,
    #'cor_reg': 1e-4,
    'cor_batch': 100,
    #"message_drop_list": [0., 0., 0.],
}

dict_map = {
    "ngcf": _ngcf,
    "lightgcn": _lightgcn,
    "dgcf": _dgcf,
    "tgcn": _tgcn,
    "kgat": _kgat,
    "dtag": _dtag,
    "disengcn": _disengcn,
    "disenhan": _disenhan,
}
