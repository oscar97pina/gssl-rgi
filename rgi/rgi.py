import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.dropout import dropout_adj

def fn_dropout(x, p, **kwargs):
    if p > 0.0:
        return F.dropout(x, p=p, **kwargs)
    return x

def fn_dropedge(edge_index, p, **kwargs):
    if p > 0.0:
        return dropout_adj(edge_index, p=p, **kwargs)[0]
    return edge_index

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def entropy_loss(x):
    B, D = x.size(0), x.size(1)
    # normalize
    x_norm = x - x.mean(dim=0)
    # variance (or std)
    #std_x = torch.sqrt(x_norm.var(dim=0) + 0.0001)
    #std_loss = torch.mean(F.relu(1 - std_x))
    # covariance
    cov_x = (x_norm.T @ x_norm) / (B - 1)
    std_loss = torch.diagonal(cov_x).add_(-1).pow_(2).sum().div(D)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(D)

    return std_loss, cov_loss

class RGI(nn.Module):
    def __init__(self, local_nn, global_nn,
                    local_pred, global_pred,
                    p_drop_x  = 0.0,
                    p_drop_u  = 0.0,
                    lambda_1=1, lambda_2=1, lambda_3=1):
        super().__init__()
        # encoders
        self.local_nn = local_nn
        self.global_nn = global_nn

        # predictors
        self.local_pred = local_pred
        self.global_pred = global_pred

        # lambdas
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3

        # dropedge
        self.p_drop_x = p_drop_x
        self.p_drop_u = p_drop_u
    
    def forward(self, x, edge_index):
        return self.local_nn(x, edge_index)
    
    def trainable_parameters(self):
        return list(self.local_nn.parameters())  + list(self.global_nn.parameters()) \
             + list(self.local_pred.parameters()) + list(self.global_pred.parameters())
    
    def local_loss(self, u, v):
        u_pred = self.local_pred(v)
        rec_loss = F.mse_loss(u, u_pred)
        var_loss, cov_loss = entropy_loss(u)
        return rec_loss, var_loss, cov_loss
    
    def global_loss(self, u, v):
        v_pred = self.global_pred(u)
        rec_loss = F.mse_loss(v, v_pred)
        var_loss, cov_loss = entropy_loss(v)
        return rec_loss, var_loss, cov_loss

    def loss(self, data):
        # get data
        x, edge_index = data.x, data.edge_index
        
        # local perturbation
        _x = fn_dropout(x, self.p_drop_x)
        _edge_index = fn_dropedge(edge_index, self.p_drop_x)       

        # local representations
        u = self.local_nn(_x, _edge_index)

        # global perturbation
        _u = fn_dropout(u, self.p_drop_u)
        _edge_index = fn_dropedge(edge_index, self.p_drop_u)  

        # global representations
        v = self.global_nn(_u, _edge_index)

        local_rec_loss, local_var_loss, local_cov_loss    = self.local_loss(u,v)
        global_rec_loss, global_var_loss, global_cov_loss = self.global_loss(u,v)

        rec_loss = local_rec_loss + global_rec_loss
        var_loss = local_var_loss + global_var_loss
        cov_loss = local_cov_loss + global_cov_loss

        loss = self.lambda_1 * rec_loss +\
               self.lambda_2 * var_loss +\
               self.lambda_3 * cov_loss
        
        logs = dict(loss = loss.item(),
                    rec_loss = rec_loss.item(),
                    var_loss = var_loss.item(),
                    cov_loss = cov_loss.item(),
                    local_rec_loss = local_rec_loss.item(),
                    local_var_loss = local_var_loss.item(),
                    local_cov_loss = local_cov_loss.item(),
                    global_rec_loss = global_rec_loss.item(),
                    global_var_loss = global_var_loss.item(),
                    global_cov_loss = global_cov_loss.item()
                    )        
        
        return loss, logs