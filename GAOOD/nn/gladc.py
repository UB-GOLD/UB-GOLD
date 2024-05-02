# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


# from torch_geometric.nn import GINConv, global_add_pool

class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = True
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = True
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):

        y = torch.matmul(adj, x)

        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias

        return y


class Encoder(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout, batch):
        super(Encoder, self).__init__()
        self.gc1 = nn.Linear(feat_size, hiddendim, bias=False)
        # self.gc2 = nn.Linear(hiddendim*2, hiddendim*2, bias=False)
        # self.gc3 = nn.Linear(hiddendim*2, hiddendim, bias=False)
        self.gc4 = nn.Linear(hiddendim, outputdim, bias=False)
        self.proj_head = nn.Sequential(nn.Linear(outputdim, outputdim), nn.ReLU(inplace=True),
                                       nn.Linear(outputdim, outputdim))
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
        self.batch = batch

    def forward(self, x, adj):
        x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        x = self.dropout(x)
        x = self.gc4(torch.matmul(adj, x))
        out, _ = torch.max(x, dim=1)
        # out = global_add_pool(x,self.batch)
        out = self.proj_head(out)

        return x, out


class attr_Decoder(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout):
        super(attr_Decoder, self).__init__()

        self.gc1 = nn.Linear(outputdim, hiddendim, bias=False)
        self.gc4 = nn.Linear(hiddendim, feat_size, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        x = self.leaky_relu(self.gc1(torch.matmul(adj, x)))
        x = self.dropout(x)

        x = self.gc4(torch.matmul(adj, x))

        return x


class stru_Decoder(nn.Module):
    def __init__(self, feat_size, outputdim, dropout):
        super(stru_Decoder, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, adj):
        x1 = x.permute(0, 2, 1)
        x = torch.matmul(x, x1)
        x = self.sigmoid(x)
        return x


class NetGe(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout, batch):
        super(NetGe, self).__init__()

        self.shared_encoder = Encoder(feat_size, hiddendim, outputdim, dropout, batch)
        self.attr_decoder = attr_Decoder(feat_size, hiddendim, outputdim, dropout)
        self.struct_decoder = stru_Decoder(feat_size, outputdim, dropout)

    def forward(self, x, adj):
        x_fake = self.attr_decoder(x, adj)
        s_fake = self.struct_decoder(x, adj)
        x2, Feat_1 = self.shared_encoder(x_fake, s_fake)

        return x_fake, s_fake, x2, Feat_1

    def l1_loss(self, input, target):
        """ L1 Loss without reduce flag.

        Args:
            input (FloatTensor): Input tensor
            target (FloatTensor): Output tensor

        Returns:
            [FloatTensor]: L1 distance between input and output
        """

        return torch.mean(torch.abs(input - target))

    ##
    def l2_loss(self, input, target, size_average=True):
        """ L2 Loss without reduce flag.

        Args:
            input (FloatTensor): Input tensor
            target (FloatTensor): Output tensor

        Returns:
            [FloatTensor]: L2 distance between input and output
        """
        if size_average:
            return torch.mean(torch.pow((input - target), 2))
        else:
            return torch.pow((input - target), 2)

    def loss_func(self, adj, A_hat, attrs, X_hat):
        # Attribute reconstruction loss
        diff_attribute = torch.pow(X_hat - attrs, 2)
        attribute_reconstruction_errors = torch.sqrt(torch.sum(diff_attribute, 1))
        attribute_cost = torch.mean(attribute_reconstruction_errors)

        # structure reconstruction loss
        diff_structure = torch.pow(A_hat - adj, 2)
        structure_reconstruction_errors = torch.sqrt(torch.sum(diff_structure, 1))
        structure_cost = torch.mean(structure_reconstruction_errors)

        return structure_cost, attribute_cost

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


class NetDe(nn.Module):
    def __init__(self, feat_size, hiddendim, outputdim, dropout, batch):
        super(NetDe, self).__init__()

        self.shared_encoder = Encoder(feat_size, hiddendim, outputdim, dropout, batch)
        self.leaky_relu = nn.LeakyReLU(0.5)
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.FloatTensor(outputdim, 1).cuda())
        init.xavier_uniform_(self.weight)
        self.m = nn.Sigmoid()

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def forward(self, x, adj):
        x, Feat = self.shared_encoder(x, adj)
        out_emb = torch.mm(Feat, self.weight)
        out_emb = self.dropout(out_emb)
        pred = self.m(out_emb)
        pred = pred.view(-1, 1).squeeze(1)

        return pred, Feat


