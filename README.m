This folder contains the code for of the paper:
RGI : Regularized Graph Infomax for self-supervised learning on graphs.

The code is based on PyTorch and PyTorch Geometric.

It is organized as follows:

1. The folder "rgi" contains the code for the RGI model and others:

- "rgi.py" RGI model.
- "data.py" utils to load the datasets.
- "models.py" implementation of the required PyTorch models (GCN, GAT, MLP and Propagation).
- "logreg.py" evaluation functions based on logistic regression.
- "scheduler.py" utils to schedule the learning rate.

2. Files to train and evaluate the models:

- "train_transductive.py" train and evaluate PPI in a transductive setting 
    (Computers, Photos, CS, Physics).
- "train_ppi.py" train RGI on the PPI dataset.
- "train_best.py" train RGI on either transductive or inductive settings employing 
    the best hyperparameters reported in the paper, stored in folder "hyperparams".

3. The folder "hyperparams" contains the best hyperparameters reported in the paper.

How to run the code:

Use train_best.py to reproduce the results. It will require the following arguments:
- dataset: the dataset to use (ppi, computers, photos, cs, physics).
- dataset_dir: the directory where the dataset is stored.
- project_dir: the directory where the results will be stored.
- seed: the random seed to use.
- num_eval_splits: the number of splits to use for evaluation.
- num_eval_epochs: frequency of evaluation
- num_workers: number of workers to use for data loading.


