import torch
import torch.nn as nn
import torch.nn.functional as F
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding

from arguments import arg_parse
from torch_geometric.transforms import Constant
import pdb

from data_loader_amp import *
import copy
from My_LA.embedding_evaluation import EmbeddingEvaluation
from My_LA.encoder import TUEncoder
from My_LA.encoder import TUEncoder_sd
from My_LA.learning import MModel
from My_LA.learning import MModel_sd
from My_LA.utils import initialize_edge_weight, initialize_node_features, set_tu_dataset_y_shape
from My_LA.LGA_learner import LGALearner
from torch_scatter import scatter
import sys
import os
import signal
import sklearn
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import interpolate
from scipy.spatial.distance import cdist
import time
from sklearn.cluster import k_means as kmeans
class simclr(nn.Module):
    def __init__(self, in_dim,hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(simclr, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(in_dim, hidden_dim, num_gc_layers)

        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                       nn.Linear(self.embedding_dim, self.embedding_dim))

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs, edge_weight=None):

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)
        y, M = self.encoder(x, edge_index, batch, edge_weight=edge_weight)
        y = self.proj_head(y)

        return y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        # sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs)+0.1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / (torch.einsum('i,j->ij', x_abs, x_aug_abs))
        sim_matrix = torch.exp((sim_matrix / T))
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss