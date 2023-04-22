import random
import numpy as np
from tqdm import tqdm

from rgi.data import get_dataset
from rgi.models import GCN, MLP, Propagate
from rgi.rgi import RGI
from rgi.logreg import *
from rgi.scheduler import CosineDecayScheduler

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import NormalizeFeatures

def seed_eth(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def run(args):
    # device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # writer
    writer = SummaryWriter(args.project_dir)

    # seed
    seed_eth(args.seed)
    
    # load data
    dataset = get_dataset(args.dataset_dir, args.dataset, transform=NormalizeFeatures())
    num_eval_splits = args.num_eval_splits

    data = dataset[0]  # all dataset include one graph
    data = data.to(device)  # permanently move in gpu memory

    # build networks
    in_size = data.x.size(1)

    local_nn = GCN( *([in_size] + \
                    (args.num_local_layers-1) * [2 * args.emb_size] + \
                     [args.emb_size])).to(device)
    global_nn  = Propagate(args.num_global_layers, method=args.shift)

    # mlp prediction networks
    local_pred   = MLP(args.emb_size, 8 * args.emb_size, args.emb_size).to(device)
    global_pred  = MLP(args.emb_size, 8 * args.emb_size, args.emb_size).to(device)

    model = RGI(local_nn, global_nn,
                local_pred, global_pred, 
                p_drop_x=args.p_drop_x, p_drop_u=args.p_drop_u,
                lambda_1=args.lambda_1, lambda_2=args.lambda_2, lambda_3=args.lambda_3).to(device)
    
    # optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler
    scheduler = CosineDecayScheduler(args.lr, args.lr_warmup_epochs, args.epochs) if args.lr_warmup_epochs is not None else None

    # train function
    def train(step):
        model.train()

        # update learning rate
        if scheduler is not None:
            lr = scheduler.get(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        optimizer.zero_grad()

        # compute loss
        loss, logs = model.loss(data)
        
        # backward & update
        loss.backward()
        optimizer.step()
        
        return logs
        
    def eval():
        model.eval()
        with torch.no_grad():
            u = model(data.x, data.edge_index).cpu().numpy()
            y = data.y.cpu().numpy()
        
        # evaluate
        val_scores, test_scores = fit_logistic_regression(u, y,
                    data_random_seed=args.seed, repeat=num_eval_splits)

        # average
        val_score = np.array(val_scores).mean()
        test_score = np.array(test_scores).mean()

        # log
        logs = {'val_score':val_score, 'test_score':test_score}

        return logs
    
    # train
    for epoch in tqdm(range(1, args.epochs + 1)):
        logs = train(epoch-1)
        # eval
        if epoch % args.num_eval_epochs == 0 or epoch == args.epochs:
            logs.update( eval() )
        # log
        for k, v in logs.items():
            writer.add_scalar(k, v, epoch)
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--project",     type=str,  default="rgi")
    parser.add_argument("--project_dir", type=str,  default="./")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--dataset", type=str,      default="cora")
    parser.add_argument("--dataset_dir", type=str,  default="./")

    parser.add_argument("--emb_size", type=int, default=512)
    parser.add_argument("--num_local_layers", type=int, default=2)
    parser.add_argument("--num_global_layers", type=int, default=2)
    parser.add_argument("--shift", type=str, default="norm_adj")

    parser.add_argument("--lambda_1", type=float, default=1)
    parser.add_argument("--lambda_2", type=float, default=1)
    parser.add_argument("--lambda_3", type=float, default=1)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--lr_warmup_epochs", type=int, default=None)
    parser.add_argument("--num_eval_epochs", type=int, default=100)
    parser.add_argument("--num_eval_splits", type=int, default=1)

    parser.add_argument("--p_drop_x",  type=float, default=0.0)
    parser.add_argument("--p_drop_u", type=float, default=0.0)

    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()
    from sklearn.utils import Bunch
    args = Bunch(**args.__dict__)
    run(args)