import random
import numpy as np
from tqdm import tqdm

from rgi.data import get_ppi, ConcatDataset
from rgi.models import GAT, Propagate, MLP, GCN
from rgi.rgi import RGI
from rgi.logreg import *
from rgi.scheduler import CosineDecayScheduler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from torch_geometric.loader import DataLoader
from torch_geometric.utils.dropout import dropout_adj

def seed_eth(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def compute_representations(model, dataset, device):
    model.eval()
    us, ys = list(), list()

    with torch.no_grad():
        for data in dataset:
            data = data.to(device)
            u = model(data.x, data.edge_index)
            us.append(u)
            ys.append(data.y)

    us = torch.cat(us, dim=0)
    ys = torch.cat(ys, dim=0)

    return [us, ys]

def run(args):
    # ppi
    args.dataset = 'ppi'
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # writer
    writer = SummaryWriter(args.project_dir)

    # seed
    seed_eth(args.seed)

    # load data
    train_dataset, val_dataset, test_dataset = get_ppi(args.dataset_dir)

    # train using both train and val splits
    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), 
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)

    # build networks
    in_size = train_dataset[0].x.size(1)

    local_nn = GAT(in_size, args.emb_size, args.emb_size)
    global_nn  = Propagate(args.num_global_layers, method=args.shift)

    # mlp prediction networks
    local_pred   = MLP(args.emb_size, 8 * args.emb_size, args.emb_size)
    global_pred  = MLP(args.emb_size, 8 * args.emb_size, args.emb_size)

    model = RGI(local_nn, global_nn,
                local_pred, global_pred, 
                p_drop_x=args.p_drop_x, p_drop_u=args.p_drop_u,
                lambda_1=args.lambda_1, lambda_2=args.lambda_2, lambda_3=args.lambda_3).to(device)
    
    print(model)

    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs)

    # train function
    def train(epoch, data):
        model.train()

        # move data to device
        data = data.to(device)

        # update learning rate
        lr = scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # forward - backward
        optimizer.zero_grad()
        loss, logs = model.loss(data)
        loss.backward()
        optimizer.step()

        return logs
        
    def eval(step):
        model.eval()

        train_data = compute_representations(model, train_dataset, device)
        val_data = compute_representations(model, val_dataset, device)
        test_data = compute_representations(model, test_dataset, device)

        val_score, test_score = ppi_train_linear_layer(train_dataset.num_classes, train_data, val_data, test_data, device)

        logs = dict(val_score=val_score, test_score=test_score)
        return logs
    
    step = 0
    for epoch in range(1, args.epochs + 1):
        ls_logs = list()
        for data in train_loader:
            logs = train(epoch, data)
            ls_logs.append(logs)
            step += 1
        
        logs = dict()
        for k in ls_logs[0].keys():
            logs[k] = sum([l[k] for l in ls_logs]) / len(ls_logs)
        logs['epoch'] = epoch
        
        if epoch % args.num_eval_epochs == 0:
            eval_logs = eval(step)
            logs.update(eval_logs)
        
        # log
        for k, v in logs.items():
            writer.add_scalar(k, v, epoch)
    
    return logs
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--project",     type=str,  default="gpc")
    parser.add_argument("--project_dir", type=str,  default="./")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset",     type=str,  default="cora")
    parser.add_argument("--dataset_dir", type=str,  default="./")

    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--num_local_layers", type=int, default=3)
    parser.add_argument("--num_global_layers", type=int, default=2)
    parser.add_argument("--shift", type=str, default="mean_adj")

    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=1)
    parser.add_argument("--lambda_3", type=float, default=1)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--num_eval_epochs", type=int, default=100)

    parser.add_argument("--p_drop_x",  type=float, default=0.0)
    parser.add_argument("--p_drop_u", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    from sklearn.utils import Bunch
    args = Bunch(**args.__dict__)
    run(args)