from tensorboardX import SummaryWriter
from utility.utils import get_config

import time
import os

CFG = get_config()
#print(f"--------------word-------------------")

class Global:
    def __init__(self) -> None:
        #self.config = get_config()
        self.pool = None
        self.out_dir = f"run/{CFG['model']}/{CFG['dataset']}/{time.strftime('%m-%d-%H-%M')}"
        self.writer = None #SummaryWriter(self.out_dir)
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

