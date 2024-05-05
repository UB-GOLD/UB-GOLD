# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
import time
from torch.autograd import Variable
import torch.nn.functional as F
from .mybase import DeepDetector
from ..nn import glocalkd
import scipy.sparse as sp
import torch.nn as nn
class GLocalKD(DeepDetector):
    def __init__(self,
                 hidden_dim=512,
                 output_dim=256,
                 num_gc_layers=3,
                 nobn = True,
                 nobias = True,
                 dropout=0.3,
                 lr=0.0001,
                 feature_dim = 53,
                 max_nodes_num = 0,
                 clip=0.1,
                 args=None,
                 num_epochs=1,
                 gpu=0,
                 **kwargs):
        super(GLocalKD, self).__init__(in_dim=None)

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gc_layers = num_gc_layers
        self.dropout = dropout
        self.lr = lr
        self.feature_dim = feature_dim
        self.max_nodes_num = max_nodes_num
        self.clip = clip
        self.args = args
        self.num_epochs = num_epochs
        self.gpu = gpu

    def process_graph(self, data):
        '''
            :param : batch_data
            :return: adj_matrixs, adj_labels, x, graph_labes
        '''
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
        '''
        :param kwargs:
        :return: GcnEncoderGraph_teacher, GcnEncoderGraph_student
        '''
        return (glocalkd.GcnEncoderGraph_teacher(input_dim=self.feature_dim,
                            hidden_dim=self.hidden_dim,
                            embedding_dim=self.output_dim,
                            label_dim=2,
                            num_layers=self.num_gc_layers,
                            args=self.args,
                            **kwargs).to(self.device)
                , glocalkd.GcnEncoderGraph_student(input_dim=self.feature_dim,
                            hidden_dim=self.hidden_dim,
                            embedding_dim=self.output_dim,
                            label_dim=2,
                            num_layers=self.num_gc_layers,
                            args=self.args,
                            **kwargs).to(self.device))


    def fit(self, dataset, args=None, label=None, dataloader=None):

        self.device = torch.device('cuda:'+str(self.gpu) if torch.cuda.is_available() else 'cpu')
        self.model_teacher, self.model_student = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model_student.parameters()), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        self.train_dataloader = dataloader
        for epoch in range(1, self.num_epochs + 1):
            total_time = 0
            total_loss = 0.0
            self.model_student.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                begin_time = time.time()
                self.model_student.zero_grad()
                adj_matrixs, adj_labels, graph_appends, _ = self.process_graph(data)
                loss = self.forward_model(adj_matrixs, adj_labels, graph_appends)
                loss.backward(loss.clone().detach())
                nn.utils.clip_grad_norm_(self.model_student.parameters(), self.clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss
                elapsed = time.time() - begin_time
                total_time += elapsed
        return True


    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        self.model_student.eval()
        loss = []
        y = []
        emb = []
        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')

        for batch_idx, data in enumerate(dataloader):
            adj_matrixs, _, graph_appends,graph_label = self.process_graph(data)
            adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
            h0 = Variable(graph_appends.float(), requires_grad=False).to(self.device)
            embed_node, embed = self.model_student(h0, adj)
            embed_teacher_node, embed_teacher = self.model_teacher(h0, adj)
            loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1)
            loss_graph = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1)
            loss_ = loss_graph + loss_node
            loss_ = np.array(loss_.cpu().detach())
            loss.append(loss_)
            if int(graph_label) != 0:
                y.append(1)
            else:
                y.append(0)
            emb.append(embed.cpu().detach().numpy())


        label_test = []
        for loss_ in loss:
            label_test.append(loss_)
        label_test = np.array(label_test)

        fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
        test_roc_ab = auc(fpr_ab, tpr_ab)
        print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
        return label_test,y


    def forward_model(self, adj_matrixs, adj_labels, graph_appends, args=None):
        adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
        h0 = Variable( graph_appends.float(), requires_grad=False).to(self.device)  # .cuda()

        embed_node, embed = self.model_student(h0, adj)
        embed_teacher_node, embed_teacher = self.model_teacher(h0, adj)
        embed_teacher = embed_teacher.detach()
        embed_teacher_node = embed_teacher_node.detach()
        loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(dim=1).mean(
            dim=0)
        loss = F.mse_loss(embed, embed_teacher, reduction='none').mean(dim=1).mean(dim=0)
        loss = loss + loss_node

        return loss
    def predict(self,
                dataset=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False,
                dataloader=None,
                args=None):
        

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score,y_all = self.decision_function(dataset, label,dataloader,args)
            output = (score,y_all)
            return output
