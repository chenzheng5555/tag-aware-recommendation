from training.early_stop import Early_stop
from train_data.abstract import Abstract_training_data
from utility.utils import printc
from utility.word import CFG

import time
import numpy as np


def epoch_training(training_data: Abstract_training_data, loss_func, opt):
    loss_list = []
    training_data.reset()
    all_loss = []
    for data in training_data.mini_batch():
        lossx = loss_func(data)
        all_loss.append([x.cpu().item() for x in lossx])
        loss = sum(lossx)
        if isinstance(opt, list):
            [op.zero_grad() for op in opt]
            loss.backward()
            [op.step() for op in opt]
        else:
            opt.zero_grad()
            loss.backward()
            opt.step()

        loss_list.append(loss.cpu().item())

    print(f"[avg_loss of each part]:{list(np.array(all_loss).sum(0))}")
    return loss_list


def add_loss_to_writer(writer, list, i, ep):
    if writer:
        n = len(list)
        for j in range(n):
            writer.add_scalar(f'Train/loss_{i}', list[j], ep * n + j)


def add_result_to_writer(writer, data_dict, epoch, name):
    if writer:
        for key, val in data_dict.items():
            if len(val) > 1:
                writer.add_scalars(f'test/{key}', \
                    {f'@{name[i]}': val[i] for i in range(len(val))}, epoch)
            else:
                writer.add_scalar(f'test/{key}', val, epoch)


class Basic_train():
    def __init__(self, train_data: list, loss_func: list, opt: list, test, args=None) -> None:
        self.train_sphase = len(train_data)
        self.train_data = train_data
        self.loss_func = loss_func
        self.opt = opt
        self.test = test
        self.early_stop = Early_stop(args)
        self.args = args

    def run(self, model):
        for ep in range(CFG['epochs']):
            model.train()
            for i in range(self.train_sphase):
                start = time.time()
                loss_list = epoch_training(self.train_data[i], self.loss_func[i], self.opt[i])
                print(f"[Epoch:{ep}][Time:{(time.time()-start)/60:.2}]:"\
                    f"avg_loss_{i} :{sum(loss_list)/len(loss_list):.5}")

                add_loss_to_writer(self.args.writer, loss_list, i, ep)

            if ep % CFG['test_interval'] == 0:
                start = time.time()
                results = self.test.run(model)
                print(f"[Epoch {ep}][Time:{(time.time()-start)/60:.2}] results: {results}")

                add_result_to_writer(self.args.writer, results, ep, CFG['topks'])

                should_stop = self.early_stop(model, results, ep)
                if should_stop == True:
                    print(f"early stop trigger at epoch {ep}")
                    break

        printc(f"best result [{self.early_stop.best_epoch}:{self.early_stop.best_result}]")
        if self.args.writer:
            self.args.writer.add_text("LOG", f"best results: epoch-{self.early_stop.best_epoch}:{self.early_stop.best_result}")
