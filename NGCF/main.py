import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim
from tensorboardX import SummaryWriter
import time

from utility import get_metrics, parse, compute_all_metric, set_seed
from data import DataBase, TrainSet, TestSet, TransTag
from model import LightGCN, NGCF, MF


def train():
    testset = TestSet(database)
    testloader = DataLoader(testset, cfg.batch_s, shuffle=True, num_workers=0)
    for epoch in range(cfg.epoch):
        trainset = TrainSet(database, cfg.neg_sample_k)
        trainloader = DataLoader(trainset, cfg.batch_t, shuffle=True, num_workers=0)
        model.train()
        all_loss = 0.0
        for i, data in tqdm(enumerate(trainloader, start=1)):
            data = [x.to(cfg.device) for x in data]
            layer_emd = model.forward(database.edge_index)
            loss = model.loss(data, layer_emd, cfg)
            opt = opt_func(model.parameters(), lr=cfg.lr)

            opt.zero_grad()
            loss.backward()
            opt.step()
            all_loss += loss.item()
            if cfg.board:
                w.add_scalar(f'Train/train loss', all_loss, epoch * int(len(trainset) / cfg.batch_t) + i)

        print(f"Epoch:{epoch}:{i} sum loss:{all_loss}")
        test(testloader, epoch)


@torch.no_grad()
def test(testloader, epoch):
    model.eval()
    metrics = []
    for _, data in tqdm(enumerate(testloader, start=1)):
        layer_emd = model.forward(database.edge_index)
        rating = model.users_rating(data, layer_emd, cfg)
        met = get_metrics(data.numpy(), database, rating.cpu(), cfg.top_k, cfg.all_user)
        metrics.append(met)
    results = compute_all_metric(metrics, cfg.top_k, cfg.all_user)
    print(f"{results}")
    if cfg.board:
        w.add_scalars(f'Test/Recall@{cfg.top_k}', {str(cfg.top_k[i]): results['recall'][i] for i in range(len(cfg.top_k))}, epoch)
        w.add_scalars(f'Test/Precision@{cfg.top_k}', {str(cfg.top_k[i]): results['precision'][i] for i in range(len(cfg.top_k))}, epoch)
        w.add_scalars(f'Test/NDCG@{cfg.top_k}', {str(cfg.top_k[i]): results['ndcg'][i] for i in range(len(cfg.top_k))}, epoch)
        w.add_scalars(f'Test/HR@{cfg.top_k}', {str(cfg.top_k[i]): results['HR'][i] for i in range(len(cfg.top_k))}, epoch)
        w.add_scalar(f'Test/auc@{cfg.top_k}', results['auc'][0], epoch)


if __name__ == "__main__":
    model_name = {"LightGCN": LightGCN, "NGCF": NGCF, 'MF': MF}
    opt_name = {"adam": optim.Adam, "rmsprop": optim.RMSprop}
    cfg = parse()
    set_seed(cfg.seed)

    cfg.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    cfg.top_k = eval(cfg.top_k)
    cfg.layer_embed = eval(cfg.layer_embed)
    database = DataBase(cfg)
    database.edge_index = torch.tensor(list(database.edge_index), dtype=torch.long).contiguous().t()
    database.edge_index = database.edge_index.to(cfg.device)

    cfg.node_list = database.node_list
    print(f"{'='*50}\n====={cfg}\n{'='*50}")

    model = model_name[cfg.model](cfg).to(cfg.device)
    opt_func = opt_name[cfg.optim]
    #for name, param in model.named_parameters():print(name,param)
    if cfg.board:
        w: SummaryWriter = SummaryWriter(f"run/{cfg.model}/{time.strftime('%m-%d-%H-%M')}")
        w.add_text('config', f'{cfg}', 0)

    train()
    if cfg.board:
        w.close()

    torch.save(model.state_dict(), "model.pth.tar")
