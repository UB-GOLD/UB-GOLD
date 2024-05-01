import torch

from .mybase import DeepDetector
from ..nn import glocalkd
from ..nn.glocalkd import GcnEncoderGraph_teacher, GcnEncoderGraph_student

import numpy as np


import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import networkx as nx
from torch_geometric.utils import to_dense_adj

import torch
import torch.nn as nn
import time
from torch.autograd import Variable
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
from sklearn.model_selection import StratifiedKFold

class GLocalKD(DeepDetector):

    def __init__(self,
                 in_dim=None,
                 datadir='dataset',
                 DS='AIDS',
                 hid_dim = 512,
                 output_dim = 256,
                 num_layers = 3,
                 dropout = 0.3,
                 lr=0.001,
                 batch_size = 300,
                 epoch = 150,
                 clip = 0.1,
                 feature = 'default',
                 max_nodes= 0,
                 nobn = True,
                 nobias = True,
                 gpu = 0,
                 bn=None,
                 max_nodes_num = 0,
                 **kwargs):
        super(GLocalKD, self).__init__(in_dim = in_dim)

        # self.in_dim = in_dim
        self.datadir = datadir
        self.DS = DS
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lr = lr
        self.batch_size = batch_size
        self.epoch = epoch
        self.clip = clip
        self.max_nodes = max_nodes
        self.nobn = nobn
        self.gpu = gpu
        self.nobias = nobias
        self.feature = feature
        self.bn = bn
        self.max_nodes_num = max_nodes_num

    def process_graph(self, data):
        num_graphs = data.ptr.size(0) - 1
        adj_matrixs = []
        adj_labels = []
        graph_appends = []
        graph_labes = []
        for i in range(num_graphs):

            graph_labes.append(int(data.y[i]))
            graph_x_size = int(data.ptr[i + 1]) - int(data.ptr[i])
            graph_x = data.x[data.ptr[i]:data.ptr[i + 1]]
            # print(graph_x.shape)
            graph_append = torch.zeros((self.max_nodes_num, graph_x.shape[1]), dtype=float)
            
            # import ipdb; ipdb.set_trace()
            
            for graph_index in range(graph_x_size):
                # print(i)
                graph_append[graph_index, :] = graph_x[graph_index, :]

            node_indices = torch.tensor(range(data.ptr[i], data.ptr[i + 1]))
            adj_matrix = torch.zeros((graph_x_size, graph_x_size), dtype=torch.float32)
            edge_indices = torch.stack(
                [data.edge_index[0][torch.where(torch.isin(data.edge_index[1], node_indices))[0]],
                 data.edge_index[1][torch.where(torch.isin(data.edge_index[1], node_indices))[0]]], dim=0)
            edge_indices = torch.sub(edge_indices, int(data.ptr[i]))
            adj_matrix[edge_indices[0], edge_indices[1]] = 1
            if self.max_nodes_num > graph_x_size:
                adj_padded = torch.zeros((self.max_nodes_num, self.max_nodes_num))
                # adj_padded = np.zeros((self.max_nodes_num , self.max_nodes_num ))
                adj_padded[:graph_x_size, :graph_x_size] = adj_matrix
                adj_label = adj_padded.numpy() + sp.eye(adj_padded.shape[0])
                adj_label = torch.tensor(adj_label)
                adj_matrixs.append(adj_padded)
                adj_labels.append(adj_label)
                graph_appends.append(graph_append)
            else:
                adj_label = adj_matrix.numpy() + sp.eye(adj_matrix.shape[0])
                adj_label = torch.tensor(adj_label)
                adj_matrixs.append(adj_matrix)
                adj_labels.append(adj_label)
                graph_appends.append(graph_append)
        adj_matrixs = torch.stack(adj_matrixs)
        adj_labels = torch.stack(adj_labels)
        graph_appends = torch.stack(graph_appends)
        graph_labes = torch.tensor(graph_labes)
        return adj_matrixs, adj_labels, graph_appends, graph_labes
        
        
    def init_model(self, **kwargs):

        return (glocalkd.GcnEncoderGraph_teacher(self.in_dim,
                                                 self.hid_dim,
                                                 self.output_dim,
                                                 2,
                                                 self.num_layers,
                                                 bn = self.bn,
                                                 **kwargs).to(self.device),
                glocalkd.GcnEncoderGraph_student(self.in_dim,
                                                 self.hid_dim,
                                                 self.output_dim,
                                                 2,
                                                 self.num_layers,
                                                 bn = self.bn,
                                                 **kwargs).to(self.device)
                )

    def fit(self, dataset, args=None, label=None, dataloader=None):
        
        print(self.max_nodes_num)
        
        if torch.cuda.is_available() and self.gpu is not None and isinstance(self.gpu, int) and self.gpu < torch.cuda.device_count():
            self.device = torch.device(f'cuda:{self.gpu}')
        else:
            self.device = torch.device('cpu')
        self.decision_score_ = None
        self.train_dataloader = dataloader

        self.model = self.init_model(**self.kwargs)
        model_teacher, model_student = self.model
        

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_student.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        epochs = []
        auroc_final = 0

        for epoch in range(1, args.num_epoch+1):    
            model_student.train()

            for data in dataloader:
                # data = data.to(self.device)
                adj_matrixs, adj_labels, graph_appends, _ = self.process_graph(data)
                model_student.zero_grad()
                loss = self.forward_model(adj_matrixs, adj_labels, graph_appends)
                loss.backward(loss.clone().detach())
                nn.utils.clip_grad_norm_(model_student.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
        return self
                
        
    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        model_teacher, model_student = self.model
        
        model_student.eval()
        y_score_all = []
        y_true_all = []
        
        for data in dataloader:
            # data = data.to(self.device)
            adj_matrixs, _, graph_appends,graph_label = self.process_graph(data)
            adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
            h0 = Variable(graph_appends.float(), requires_grad=False).to(self.device)
            # import ipdb; ipdb.set_trace()
            embed_node, embed = model_student(h0, adj)
            embed_teacher_node, embed_teacher = model_teacher(h0, adj)
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
            loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
            loss_ = loss_graph + loss_node
            
            
            # modified_labels = [1 if x != 0 else 0 for x in graph_label]
            # if int(graph_label) != 0:
            #     y_true_all.append(1)
            # else:
            #     y_true_all.append(0)
            # y_true_all.append(modified_labels)
            y_true_all = y_true_all + graph_label.detach().cpu().tolist()
            y_score_all = y_score_all + loss_.detach().cpu().tolist()
            # y_score_all.append(loss_.detach().cpu())
                               
        return y_score_all, y_true_all
            
            # embed_node, embed = model_student(data.x, data.adj)
            # embed_teacher_node, embed_teacher = model_teacher(data.x, data.adj)
            # import ipdb; ipdb.set_trace()
            # loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=0).mean(dim=0)
            # loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=0)
            # loss_ = loss_graph + loss_node
            # loss_ = np.array(loss_.cpu().detach())
            
            # y_score_all = y_score_all + loss_.detach().cpu().tolist()
            # y_true = data.y
            # y_true_all = y_true_all + y_true.detach().cpu().tolist()        


        # return y_score_all,y_true_all
            
    def forward_model(self, adj_matrixs,adj_labels, graph_appends, args=None):
        model_teacher, model_student = self.model
        adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
        h0 = Variable( graph_appends.float(), requires_grad=False).to(self.device)  # .cuda()
        adj_label = Variable(adj_labels.float(), requires_grad=False).to(self.device)  # .cuda()
        
        embed_node, embed = model_student(h0, adj)
        embed_teacher_node, embed_teacher = model_teacher(h0, adj)
        embed_teacher =  embed_teacher.detach()
        embed_teacher_node = embed_teacher_node.detach()
        loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
        loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
        loss = loss + loss_node
        # dataset = self.process_graph(dataset)
        # model_teacher, model_student = self.model
        # embed_node, embed = model_student(dataset.x, dataset.adj)
        # embed_teacher_node, embed_teacher = model_teacher(dataset.x, dataset.adj)
        # embed_teacher = embed_teacher.detach()
        # embed_teacher_node = embed_teacher_node.detach()
        # loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=1).mean(dim=0)
        # loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=0)
        # loss = loss + loss_node

        return loss

    def predict(self,
            dataset=None,
            label=None,
            return_pred=True,
            return_score=False,
            return_prob=False,
            return_conf=False,
            return_emb=False,
            dataloader=None,
            args=None):

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score,y_all = self.decision_function(dataset, label, dataloader, args)
            output = (score,y_all)
            return output

