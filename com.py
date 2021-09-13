from utility.word import CFG

from torch import optim
from data import *
from model import *
from train_data import *
from training import *


def ngcf_comp(args):
    data = TGCN_load(args)
    model = NGCF(data).to(CFG['device'])
    train_data = [BPR_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


def lightgcn_comp(args):
    data = TGCN_load(args)
    model = LightGCN(data).to(CFG['device'])
    train_data = [BPR_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


def dgcf_comp(args):
    data = TGCN_load(args)
    model = DGCF(data).to(CFG['device'])
    train_data = [DGCF_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


def disengcn_comp(args):
    data = TGCN_load(args)
    model = DisenGCN(data).to(CFG['device'])
    train_data = [DGCF_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


def disenhan_comp(args):
    data = TGCN_load(args)
    model = DisenHAN(data).to(CFG['device'])
    train_data = [DGCF_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


def tgcn_comp(args):
    data = TGCN_load(args)
    model = TGCN(data).to(CFG['device'])
    train_data = [BPR_training_data(data, args), TransTag_training_data(data, args)]
    opt = optim.Adam(model.parameters(), lr=CFG['lr'])
    opts = [opt, opt]
    loss_func = [model.loss, model.transtag_loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opts, test, args)
    return model, train, test


def kgat_comp(args):
    data = TGCN_load(args)
    model = KGAT(data).to(CFG['device'])
    train_data = [BPR_training_data(data, args), KGAT_training_data(data, args)]
    opt = optim.Adam(model.parameters(), lr=CFG['lr'])
    opts = [opt, opt]
    loss_func = [model.loss, model.transe_loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opts, test, args)
    return model, train, test


_dtag_map = {
    'dtag': DTAG,
    'dtag1': DTAG1,
    'dtag2': DTAG2,
    'dtag3': DTAG3,
    'dtag4': DTAG4,
    'dtag5': DTAG5,
    'dtag6': DTAG6,
    'dtag7': DTAG7,
}


def dtag_comp(args):
    data = TGCN_load(args)
    model = _dtag_map[CFG['model']](data).to(CFG['device'])
    train_data = [DGCF_training_data(data, args)]
    opt = [optim.Adam(model.parameters(), lr=CFG['lr'])]
    loss_func = [model.loss]
    test = Basic_test(data, args)
    train = Basic_train(train_data, loss_func, opt, test, args)
    return model, train, test


model_dict = {
    'ngcf': ngcf_comp,
    'lightgcn': lightgcn_comp,
    'tgcn': tgcn_comp,
    'kgat': kgat_comp,
    'dgcf': dgcf_comp,
    'disengcn': disengcn_comp,
    'disenhan': disenhan_comp,
    'dtag': dtag_comp,
}
