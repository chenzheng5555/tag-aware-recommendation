import numpy as np
import torch
import random
import multiprocessing

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def multi_core():
    pass