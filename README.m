# Feature Propagation as Self-Supervision Signals on Graphs

This repository contains the PyTorch implementation of the "Feature propagation as self-supervision signals on graphs" paper for self-supervised learning on graphs. We introduce Regularized Graph Infomax (RGI), designed to address the limitations of current algorithms that rely on contrastive learning and invariance assumptions under graph augmentations for self-supervised learning on graphs. Instead, RGI uses feature propagation to obtain the self-supervision signals.

## Folder Structure

1. **`rgi` Folder:**
    - `rgi.py`: Implementation of the RGI model.
    - `data.py`: Utilities for loading datasets.
    - `models.py`: Implementation of required PyTorch models (GCN, GAT, MLP, and Propagation).
    - `logreg.py`: Evaluation functions based on logistic regression.
    - `scheduler.py`: Utilities for scheduling the learning rate.

2. **Training and Evaluation Files:**
    - `train_transductive.py`: Train and evaluate RGI on PPI in a transductive setting (Computers, Photos, CS, Physics).
    - `train_ppi.py`: Train RGI on the PPI dataset.
    - `train_best.py`: Train RGI on either transductive or inductive settings using the best hyperparameters reported in the paper. Hyperparameters are stored in the "hyperparams" folder.

3. **`hyperparams` Folder:**
    - Contains the best hyperparameters reported in the paper.

## How to Run the Code

To reproduce the results, use `train_best.py` with the following arguments:

- `dataset`: The dataset to use (ppi, computers, photos, cs, physics).
- `dataset_dir`: The directory where the dataset is stored.
- `project_dir`: The directory where the results will be stored.
- `seed`: The random seed to use.
- `num_eval_splits`: The number of splits to use for evaluation.
- `num_eval_epochs`: Frequency of evaluation.
- `num_workers`: Number of workers to use for data loading.

Alternatively, you can use `train_transductive.py` to train and evaluate RGI on Computers, Photos, CS, Physics or `train_ppi.py` for the PPI dataset with your own hyperparameters.

## Dependencies

The code is based on PyTorch and PyTorch Geometric. Make sure to install the required dependencies before running the code.

## Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@article{PINA2024111512,
title = {Feature propagation as self-supervision signals on graphs},
journal = {Knowledge-Based Systems},
volume = {289},
pages = {111512},
year = {2024},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2024.111512},
url = {https://www.sciencedirect.com/science/article/pii/S0950705124001473},
author = {Oscar Pina and Verónica Vilaplana},
keywords = {Graph neural network, Self-supervised learning, Graph representation learning, Regularization},
abstract = {Self-supervised learning is gaining considerable attention as a solution to avoid the requirement of extensive annotations in representation learning on graphs. Current algorithms are based on contrastive learning, which is computationally and memory expensive, and the assumption of invariance under certain graph augmentations. However, graph transformations such as edge sampling may modify the semantics of the data, potentially leading to inaccurate invariance assumptions. To address these limitations, we introduce Regularized Graph Infomax (RGI), a simple yet effective framework for node level self-supervised learning that trains a graph neural network encoder by maximizing the mutual information between the output node embeddings and their propagation through the graph, which encode the nodes’ local and global context, respectively. RGI generates self-supervision signals through feature propagation rather than relying on graph data augmentations. Furthermore, the method is non-contrastive and does not depend on a two branch architecture. We run RGI on both transductive and inductive settings with popular graph benchmarks and show that it can achieve state-of-the-art performance regardless of its simplicity.}
}