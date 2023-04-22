import yaml
import argparse
import train_ppi, train_transductive
from sklearn.utils import Bunch

BEST_DIR = "./hyperparams/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='computers')
    parser.add_argument('--project_dir', type=str, default="./")
    parser.add_argument('--dataset_dir', type=str, default=2000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_eval_splits', type=int, default=20)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--num_eval_epochs', type=int, default=1000)

    args = parser.parse_args()
    args = Bunch(**args.__dict__)

    # load best hyperparams
    with open(BEST_DIR + args.dataset + ".yaml", 'r') as f:
        best = yaml.load(f, Loader=yaml.FullLoader)
    args.update(best)

    if args.dataset == 'ppi':
        train_ppi.run(args)
    else:
        train_transductive.run(args)