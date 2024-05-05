# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 17:16:36 2021

@author: MaRongrong
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False,
                 dropout=0.0, bias=True):
        super(GraphConv, self).__init__()
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None

    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        y = torch.matmul(adj, x)
        if self.add_self:
            y += x
        y = torch.matmul(y, self.weight)
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y


class GcnEncoderGraph_teacher(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=False, bn=True, dropout=0.0, args=None):
        super(GcnEncoderGraph_teacher, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)  # relu
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        return x, output


class GcnEncoderGraph_student(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers,
                 pred_hidden_dims=[], concat=False, bn=True, dropout=0.1, args=None):
        super(GcnEncoderGraph_student, self).__init__()
        self.concat = concat
        add_self = not concat
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(
            input_dim, hidden_dim, embedding_dim, num_layers,
            add_self, normalize=True, dropout=dropout)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self,
                          normalize=False, dropout=0.0):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self,
                               normalize_embedding=normalize, bias=self.bias)
        conv_block = nn.ModuleList(
            [GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self,
                       normalize_embedding=normalize, dropout=dropout, bias=self.bias)
             for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self,
                              normalize_embedding=normalize, bias=self.bias)
        return conv_first, conv_block, conv_last

    def apply_bn(self, x):
        ''' Batch normalization of 3D tensor x
        '''
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        ''' Perform forward prop with graph convolution.
        Returns:
            Embedding matrix with dimension [batch_size x num_nodes x embedding]
        '''

        x = conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        for i in range(len(conv_block)):
            x = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
        x = conv_last(x, adj)
        x_all.append(x)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # conv
        x = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        for i in range(self.num_layers - 2):
            x = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
        x = self.conv_last(x, adj)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        return x, output
