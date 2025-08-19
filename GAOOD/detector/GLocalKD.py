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
import os
from GAOOD.metric import *
class GLocalKD(DeepDetector):
    def __init__(self,
                 args=None,
                 **kwargs):
        super(GLocalKD, self).__init__(in_dim=None)

        self.args = args
        self.build_save_path()

    def build_save_path(self):
        print(self.args)
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.args.exp_type == 'oodd':
            path = os.path.join(path, 'model_save',self.args.model, self.args.exp_type, self.args.DS_pair)
        elif self.args.DS.startswith('Tox21'):
            path = os.path.join(path, 'model_save', self.args.model, self.args.exp_type+'Tox21', self.args.DS)
        else:
            path = os.path.join(path, 'model_save',self.args.model, self.args.exp_type, self.args.DS)
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.delete_files_in_directory(path)

    def delete_files_in_directory(self, directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                self.delete_files_in_directory(file_path)


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
        return (glocalkd.GcnEncoderGraph_teacher(input_dim=self.args.dataset_num_features,
                            hidden_dim=self.args.hidden_dim,
                            embedding_dim=self.args.output_dim,
                            label_dim=2,
                            num_layers=self.args.num_layer,
                            bn = self.args.bn,
                            args=self.args,
                            **kwargs).to(self.device)
                , glocalkd.GcnEncoderGraph_student(input_dim=self.args.dataset_num_features,
                            hidden_dim=self.args.hidden_dim,
                            embedding_dim=self.args.output_dim,
                            label_dim=2,
                            num_layers=self.args.num_layer,
                            bn = self.args.bn,
                            args=self.args,
                            **kwargs).to(self.device))


    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        self.max_nodes_num = args.max_nodes_num

        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model_teacher, self.model_student = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model_student.parameters()), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        self.max_AUC = 0
        
        stop_counter = 0  
        N = 30  

        preprocessed = []
        for batch_idx, data in enumerate(dataloader):
            adj, _, h0, _ = self.process_graph(data)
            preprocessed.append(tuple([adj, h0]))

        preprocessed_val = []
        for batch_idx, data in enumerate(dataloader_val):
            adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)
            preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))

        for epoch in range(1, args.num_epoch + 1):
            total_time = 0
            total_loss = 0.0
            self.model_student.train()
            for batch_idx, data in enumerate(dataloader):
                begin_time = time.time()
                self.model_student.zero_grad()
                # adj, _, h0, _ = self.process_graph(data)
                adj, h0 = preprocessed[batch_idx]
                loss = self.forward_model(adj, h0)
                loss.backward(loss.clone().detach())
                nn.utils.clip_grad_norm_(self.model_student.parameters(), args.clip)
                optimizer.step()
                scheduler.step()
                total_loss += loss
                elapsed = time.time() - begin_time
                total_time += elapsed
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model_student.eval()
                loss = []
                y = []
                emb = []

                for adj_matrixs, graph_appends, graph_label in preprocessed_val:
                    # adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)

                    adj = Variable(adj_matrixs.float(), requires_grad=False).to('cuda')  # .cuda()
                    h0 = Variable(graph_appends.float(), requires_grad=False).to('cuda')
                    # adj = Variable(data['adj'].float(), requires_grad=False).cuda()
                    # h0 = Variable(data['feats'].float(), requires_grad=False).cuda()

                    embed_node, embed = self.model_student(h0, adj)
                    embed_teacher_node, embed_teacher = self.model_teacher(h0, adj)
                    loss_node = torch.mean(F.mse_loss(embed_node, embed_teacher_node, reduction='none'), dim=2).mean(
                        dim=1)
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
                val_auc = ood_auc(y, label_test)
                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    stop_counter = 0  
                    torch.save(self.model_teacher, os.path.join(self.path, 'model_teacher.pth'))
                    torch.save(self.model_student, os.path.join(self.path, 'model_student.pth'))
                else:
                    stop_counter += 1  
                print('[TRAIN] Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, self.max_AUC))
                if stop_counter >= N:
                    print(f'Early stopping triggered after {epoch} epochs due to no improvement in AUC for {N} consecutive evaluations.')
                    break  
                    
        return True
    def is_directory_empty(self,directory):
        files_and_dirs = os.listdir(directory)
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        if self.is_directory_empty(self.path):
            pass
        else:
            self.model_teacher = torch.load(os.path.join(self.path,'model_teacher.pth'))
            self.model_student = torch.load(os.path.join(self.path, 'model_student.pth'))

        self.model_student.eval()
        loss = []
        y = []
        emb = []
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

        for batch_idx, data in enumerate(dataloader):
            adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)
            preprocessed.append(tuple([adj_matrixs, graph_appends, graph_label]))

        # for batch_idx, data in enumerate(dataloader):
        for adj_matrixs,  graph_appends,graph_label in preprocessed:
            # adj_matrixs, _, graph_appends,graph_label = self.process_graph(data)
            adj = Variable(adj_matrixs.float(), requires_grad=False).to('cuda')  # .cuda()
            h0 = Variable(graph_appends.float(), requires_grad=False).to('cuda')
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

        # fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
        # test_roc_ab = auc(fpr_ab, tpr_ab)
        # print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
        return label_test,y


    def forward_model(self, adj, h0):
        adj = Variable(adj.float(), requires_grad=False).to('cuda')
        h0 = Variable(h0.float(), requires_grad=False).to('cuda')

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
