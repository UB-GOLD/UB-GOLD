import torch
import numpy as np
import random
from texttable import Texttable
import os
from torch_geometric.utils import to_scipy_sparse_matrix

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def args_print(args, logger):
    _dict = vars(args)
    table = Texttable()
    table.add_row(["Parameter", "Value"])
    for k in _dict:
        table.add_row([k, _dict[k]])
    logger.info(table.draw())

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def propagate(feature, edge_index, order):
    x = feature
    y = feature
    A = sparse_mx_to_torch_sparse_tensor(to_scipy_sparse_matrix(edge_index, num_nodes=x.shape[0])).to(x.device)
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        y.add_(x)
    return y.div_(order+1.0).detach_()

def rand_prop(features, edge_index, order, dropnode, training):
    """For robust GRAND baseline"""
    n = features.shape[0]
    drop_rates = torch.FloatTensor(np.ones(n) * dropnode)
    if training:
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1).to(features.device)
        features = masks * features
    else:
        features = features * (1. - dropnode)
    features = propagate(features, edge_index, order)
    return features

def consis_loss(logps, temp):
    """For robust GRAND baseline"""
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)

    sharp_p = (torch.pow(avg_p, 1. / temp) / torch.sum(torch.pow(avg_p, 1. / temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p - sharp_p).pow(2).sum(1))
    loss = loss / len(ps)
    return loss

