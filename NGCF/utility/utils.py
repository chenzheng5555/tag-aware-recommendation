import random
import torch
import numpy as np
import argparse
import time
from collections import Iterable
from utility.config import dict_map


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="parse commend args")
    parser.add_argument('--model', type=str, default="dtag6", help="model")
    parser.add_argument('--data_root', type=str, default=r"C:\data", help="root data path")
    parser.add_argument('--dataset', type=str, default='movielens2', help='data set')

    parser.add_argument('--train_batch', type=int, default=512, help="the batch size for bpr loss training procedure")
    parser.add_argument('--test_batch', type=int, default=512, help="the batch size of users for testing")
    parser.add_argument('--has_val', type=bool, default=False, help='')
    parser.add_argument('--use_tag', type=bool, default=True, help='')
    parser.add_argument('--patient_epoch', type=int, default=10, help="the patient epochs of early stop")
    parser.add_argument('--test_interval', type=int, default=5, help="interval epochs of print test result")
    parser.add_argument('--early_stop_key', type=str, default='ndcg', help='used by early stop')

    parser.add_argument('--topks', nargs='?', default="[10,20]", help="@k test list")
    parser.add_argument('--lr', type=float, default=0.01, help="the learning rate")
    parser.add_argument('--reg', type=float, default=0, help="the learning rate")
    parser.add_argument('--cor_reg', type=float, default=0, help="the learning rate")

    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--dim_latent', type=int, default=64, help="the latent embedding size of the model")
    parser.add_argument('--dim_layer_list', nargs='?', default="[64,32,16]", help="embedding size of each layer")
    parser.add_argument('--message_drop_list', nargs='?', default="[0.,0.,0.]", help="")
    parser.add_argument('--node_drop', type=float, default=0., help="")

    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--cpu_core', type=int, default=4, help="cpu cores")
    parser.add_argument('--split_adj_k', type=int, default=1, help="split sparse adj k fold")

    return parser.parse_args()


def get_config():
    config = {}
    args = parse_args()
    args.topks = eval(args.topks)
    args.dim_layer_list = eval(args.dim_layer_list)
    args.message_drop_list = eval(args.message_drop_list)
    config.update(vars(args))
    config["device"] = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = config['model']
    if config['model'][:4] == 'dtag':
        model = 'dtag'
    config.update(dict_map[model])
    return config


def toname(*args):
    name = ""
    for x in args:
        if isinstance(x, Iterable):
            name += '[' + toname(*x) + '], '
        else:
            name += x.__class__.__name__ + ', '

    return name


def printc(args):
    print(f"\033[0;33;40m{args}\033[0m")
