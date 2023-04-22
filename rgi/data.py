"""
Based on https://github.com/nerdslab/bgrl
"""

import numpy as np
import torch

from torch_geometric import datasets
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected

def get_dataset(root, name, transform=NormalizeFeatures()):
    pyg_dataset_dict = {
        'cora' : (datasets.Planetoid, 'Cora'),
        'citeseer' : (datasets.Planetoid, 'CiteSeer'),
        'pubmed' : (datasets.Planetoid, 'PubMed'),
        'cs': (datasets.Coauthor, 'CS'),
        'physics': (datasets.Coauthor, 'physics'),
        'computers': (datasets.Amazon, 'Computers'),
        'photos': (datasets.Amazon, 'Photo'),
    }

    assert name in pyg_dataset_dict, "Dataset must be in {}".format(list(pyg_dataset_dict.keys()))

    dataset_class, name = pyg_dataset_dict[name]
    dataset = dataset_class(root, name=name, transform=transform)

    return dataset

def get_ppi(root, transform=None):
    train_dataset = datasets.PPI(root, split='train', transform=transform)
    val_dataset   = datasets.PPI(root, split='val',   transform=transform)
    test_dataset  = datasets.PPI(root, split='test',  transform=transform)
    return train_dataset, val_dataset, test_dataset


class ConcatDataset(InMemoryDataset):
    r"""
    PyG Dataset class for merging multiple Dataset objects into one.
    """
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        self.__indices__ = None
        self.__data_list__ = []
        for dataset in datasets:
            self.__data_list__.extend(list(dataset))
        self.data, self.slices = self.collate(self.__data_list__)