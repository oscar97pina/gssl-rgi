import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GCNConv, GATConv
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor, matmul

class GCN(nn.Module):
    def __init__(self, *dims):
        super().__init__()

        convs, acts, norms = list(), list(), list()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            convs.append( GCNConv(in_dim, out_dim) )
            if i == len(dims) - 2: # -2 since from iterating len-1 values
                acts.append(nn.Identity())
                norms.append(nn.Identity())
            else:
                acts.append( nn.ReLU() )
                norms.append(BatchNorm(out_dim))
        
        self.convs = nn.ModuleList(convs)
        self.acts  = nn.ModuleList(acts)
        self.norms = nn.ModuleList(norms)

    def forward(self, x, edge_index):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
            x = self.norms[i](x)
            x = self.acts[i](x)
        return x

class GAT(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=512, num_heads=4):
        """
        Extracted from https://github.com/nerdslab/bgrl
        """

        super().__init__()

        self._conv1 = GATConv(input_size, hidden_size, heads=num_heads, concat=True)
        self._skip1 = nn.Linear(input_size, num_heads * hidden_size)

        self._conv2 = GATConv(num_heads * hidden_size, hidden_size, heads=num_heads, concat=True)
        self._skip2 = nn.Linear(num_heads * hidden_size, num_heads * hidden_size)

        self._conv3 = GATConv(num_heads * hidden_size, output_size, heads=num_heads+2, concat=False)
        self._skip3 = nn.Linear(num_heads * hidden_size, output_size)

    def forward(self, x, edge_index, batch=None):
        x = F.elu(self._conv1(x, edge_index) + self._skip1(x))
        x = F.elu(self._conv2(x, edge_index) + self._skip2(x))
        x = F.elu(self._conv3(x, edge_index) + self._skip3(x))

        return x

class MLP(nn.Sequential):
    def __init__(self, *dims):
        mls = list()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            mls.append(nn.Linear(in_dim, out_dim))
            if i < len(dims) - 2:
                mls.append(nn.ReLU())
        super().__init__(*mls)

class Propagate(nn.Module):
    def __init__(self, K, method='norm_adj'):
        super().__init__()
        self.K = K

        if method == 'norm_adj':
            self.prop = NormAdjPropagate(K)
        elif method == 'norm_lap':
            self.prop = NormLapPropagate(K)
        elif method == 'mean_adj':
            self.prop = MeanAdjPropagate(K)
        else:
            raise ValueError(f'Unknown propagation method: {method}')
    
    def forward(self, x, edge_index, batch=None):
        return self.prop(x, edge_index, batch=batch)
    
    def __repr__(self):
        return f'{self.__class__.__name__}({self.K}, {self.prop.__class__.__name__})'

class NormLapPropagate(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
    
    def _compute_adj_t(self, num_nodes, edge_index):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        return adj_t

    def forward(self, x, edge_index, batch=None):
        adj_t = self._compute_adj_t(x.size(0), edge_index)
        for i in range(self.K):
            x = x - adj_t @ x
        return x

class NormAdjPropagate(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
    
    def _compute_adj_t(self, num_nodes, edge_index):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))

        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        return adj_t

    def forward(self, x, edge_index, batch=None):
        adj_t = self._compute_adj_t(x.size(0), edge_index)
        for i in range(self.K):
            x = adj_t @ x
        return x

class MeanAdjPropagate(nn.Module):
    def __init__(self, K):
        super().__init__()
        self.K = K
    def _compute_adj_t(self, num_nodes, edge_index):
        row, col = edge_index
        adj_t = SparseTensor(row=col, col=row,
                             sparse_sizes=(num_nodes, num_nodes))
        return adj_t
    def forward(self, x, edge_index, batch=None):
        adj_t = self._compute_adj_t(x.size(0), edge_index)
        for i in range(self.K):
            x = matmul(adj_t, x, reduce='mean')       
        return x