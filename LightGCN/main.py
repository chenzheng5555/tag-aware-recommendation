import torch
import numpy as np
from tqdm import tqdm
from model import LightGCN
from torch import optim
from data import DataBase, TrainSet
from utils import *
from torch.utils.data import DataLoader
from parse import parse_args
import multiprocessing

def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in args.topks:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(NDCGatK_r(groundTrue,r,k))
    return {'recall':np.array(recall),
            'precision':np.array(pre),
            'ndcg':np.array(ndcg)}

@torch.no_grad()
def test():
    max_k=max(args.topks)
    results = {'precision': np.zeros(len(args.topks)),
               'recall': np.zeros(len(args.topks)),
               'ndcg': np.zeros(len(args.topks))}
    users_list, rating_list = [], []
    groundTrue_list=[]
    users=list(database.userTest.keys())
    for batch_users in minibatch(users,args.test_batch):
        allpos = [database.userPos[u] for u in batch_users]
        groundTrue=[database.userTest[u] for u in batch_users]
        batch_users_gpu=torch.tensor(batch_users,dtype=torch.long,device=args.device)
        rating =model.getUsersRating(batch_users_gpu)

        exclude_index=[]
        exclude_items=[]
        for range_i,items in enumerate(allpos):
            exclude_index.extend([range_i]*len(items))
            exclude_items.extend(items)
        rating[exclude_index,exclude_items]=-(1<<10)
        _,rating_K=torch.topk(rating,k=max_k)
        users_list.append(batch_users)
        rating_list.append(rating_K.cpu())
        groundTrue_list.append(groundTrue)

    X=zip(rating_list,groundTrue_list)
    pool = multiprocessing.Pool(args.CORES)
    pre_results = pool.map(test_one_batch, X)
    pool.close()
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    results['recall'] /= float(len(users))
    results['precision'] /= float(len(users))
    results['ndcg'] /= float(len(users))

    print(results)


def train():
    opt = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        traindata = TrainSet(database)
        dataloader = DataLoader(traindata, args.train_batch, shuffle=True, num_workers=0)
        model.train()
        aver_loss=0.
        for i, data in tqdm(enumerate(dataloader, start=0)):
            users, pos, neg = data[:, 0], data[:, 1], data[:, 2]
            loss, reg_loss = model.getBPRLoss(users, pos, neg)
            loss = loss + args.decay * reg_loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            aver_loss += loss.cpu().item()

        print(f"[epoch:{epoch}]-aver_loss:{aver_loss}")

        model.eval()
        if epoch%20==0:
            test()


args = parse_args()
args.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
args.topks=eval(args.topks)
args.CORES = 6

if __name__ == "__main__":
    print(args)
    database=DataBase(args)
    model = LightGCN(database, args).to(args.device)
    train()
