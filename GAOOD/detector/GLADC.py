# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from .mybase import DeepDetector
from ..nn import gladc
import scipy.sparse as sp
import os
from GAOOD.metric import *
class GLADC(DeepDetector):
    def __init__(self,
                 max_nodes=0,
                 num_epochs=1,
                 batch_size=300,
                 hidden_dim=256,
                 output_dim=128,
                 num_gc_layers=2,
                 bn = None,
                 dropout=0.1,
                 lr=0.0001,
                 feature_dim = 53,
                 max_nodes_num = 0,
                 DS = 'BZR',
                 DS_pair = None,
                 exp_type = None,
                 model_name = None,
                 args = None,
                 **kwargs):
        super(GLADC, self).__init__(in_dim=None)

        self.max_nodes = max_nodes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gc_layers = num_gc_layers
        self.dropout = dropout
        self.lr = lr
        self.feature_dim = feature_dim
        self.max_nodes_num = max_nodes_num
        self.DS = DS
        self.DS_pair = DS_pair
        self.exp_type = exp_type
        self.model_name = model_name
        self.bn = bn
        self.args = args,
        self.build_save_path()

    def build_save_path(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.exp_type == 'oodd':
            path = os.path.join(path, 'model_save',self.model_name, self.exp_type, self.DS_pair)
        elif self.DS.startswith('Tox21'):
            path = os.path.join(path, 'model_save', self.model_name, self.exp_type+'Tox21', self.DS)
        else:
            path = os.path.join(path, 'model_save',self.model_name, self.exp_type, self.DS)
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
            graph_append = torch.zeros((self.max_nodes_num, graph_x.shape[1]), dtype=float)
            for graph_index in range(graph_x_size):
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
        :return: NetGe, noise_NetG
        '''
        return (gladc.NetGe(feat_size=self.feature_dim,
                            hiddendim=self.hidden_dim,
                            outputdim=self.output_dim,
                            dropout=self.dropout,
                            batch=self.batch_size,
                            **kwargs).to(self.device)
                ,gladc.Encoder(feat_size=self.feature_dim,
                                hiddendim=self.hidden_dim,
                                outputdim=self.output_dim,
                                dropout=self.dropout,
                                batch=self.batch_size,
                                **kwargs).to(self.device))

    def gen_ran_output(self, h0, adj, model, vice_model):

        for (adv_name, adv_param), (name, param) in zip(vice_model.named_parameters(), model.named_parameters()):
            # print(name)
            # print(param.data.std())
            if name.split('.')[0] == 'proj_head':
                adv_param.data = param.data
            else:
                adv_param.data = param.data + 1.0 * torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(
                    self.device)
        x1_r, Feat_0 = vice_model(h0, adj)
        return x1_r, Feat_0

    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        # path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # path = os.path.join(path, 'model_save', self.model_name, self.DS)
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.NetGe, self.noise_NetG = self.init_model(**self.kwargs)
        optimizerG = torch.optim.Adam(self.NetGe.parameters(), lr=self.lr)
        self.max_AUC = 0

        stop_counter = 0  
        N = 30  
        
        preprocessed = []
        for batch_idx, data in enumerate(dataloader):
            adj_matrixs, adj_labels, graph_appends, _ = self.process_graph(data)
            preprocessed.append(tuple([adj_matrixs, adj_labels, graph_appends]))

        preprocessed_val = []
        for batch_idx, data in enumerate(dataloader_val):
            adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)
            preprocessed_val.append(tuple([adj_matrixs, graph_appends, graph_label]))
        
        for epoch in range(1, self.num_epochs + 1):
            total_lossG = 0.0
            self.NetGe.train()
            for adj_matrixs, adj_labels, graph_appends in preprocessed:
                # adj_matrixs, adj_labels, graph_appends, _ = self.process_graph(data)
                lossG = self.forward_model(adj_matrixs, adj_labels, graph_appends)
                optimizerG.zero_grad()
                lossG.backward()
                optimizerG.step()
                total_lossG += lossG
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.NetGe.eval()
                loss = []
                y = []

                for adj_matrixs, graph_appends, graph_label in preprocessed_val:

                    # adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)
                    adj = Variable(adj_matrixs.float().float(), requires_grad=False).to(self.device)
                    h0 = Variable(graph_appends.float(), requires_grad=False).to(self.device)

                    x1_r, Feat_0 = self.NetGe.shared_encoder(h0, adj)
                    x_fake, s_fake, x2, Feat_1 = self.NetGe(x1_r, adj)
                    loss_node = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
                    loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
                    loss_ = loss_node + loss_graph
                    loss_ = np.array(loss_.cpu().detach())
                    loss.append(loss_)
                    if int(graph_label) != 0:
                        y.append(1)
                    else:
                        y.append(0)
                label_test = []
                for loss_ in loss:
                    label_test.append(loss_)
                label_test = np.array(label_test)
                val_auc = ood_auc(y, label_test)

                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    stop_counter = 0  
                    torch.save(self.NetGe, os.path.join(self.path, 'model_NetGe.pth'))
                    torch.save(self.noise_NetG, os.path.join(self.path, 'model_noise_NetG.pth'))
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
        # path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # path = os.path.join(path, 'model_save', self.model_name, self.DS)
        if self.is_directory_empty(self.path):
            print("Can't find the path")
        else:
            print("Loading Model Weight")
            self.NetGe = torch.load(os.path.join(self.path,'model_NetGe.pth'))
        self.NetGe.eval()
        loss = []
        y = []
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

        preprocessed = []
        for batch_idx, data in enumerate(dataloader):
            adj_matrixs, _, graph_appends, graph_label = self.process_graph(data)
            preprocessed.append(tuple([adj_matrixs, graph_appends, graph_label]))

        for adj_matrixs,  graph_appends,graph_label in preprocessed:
            # adj_matrixs, _, graph_appends,graph_label = self.process_graph(data)
            adj = Variable(adj_matrixs.float().float(), requires_grad=False).to(self.device)
            h0 = Variable(graph_appends.float(), requires_grad=False).to(self.device)
            x1_r, Feat_0 = self.NetGe.shared_encoder(h0, adj)
            x_fake, s_fake, x2, Feat_1 = self.NetGe(x1_r, adj)
            loss_node = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
            loss_graph = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1)
            loss_ = loss_node + loss_graph
            loss_ = np.array(loss_.cpu().detach())
            loss.append(loss_)
            if int(graph_label) != 0:
                y.append(1)
            else:
                y.append(0)
        label_test = []
        for loss_ in loss:
            label_test.append(loss_)
        label_test = np.array(label_test)
        return label_test,y


    def forward_model(self, adj_matrixs, adj_labels, graph_appends, args=None):
        adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
        h0 = Variable( graph_appends.float(), requires_grad=False).to(self.device)  # .cuda()
        adj_label = Variable(adj_labels.float(), requires_grad=False).to(self.device)  # .cuda()

        x1_r, Feat_0 = self.NetGe.shared_encoder(h0, adj)
        # print(x1_r.shape)
        # print(Feat_0.shape)
        x1_r_1, Feat_0_1 = self.gen_ran_output(h0, adj, self.NetGe.shared_encoder, self.noise_NetG)
        x_fake, s_fake, x2, Feat_1 = self.NetGe(x1_r, adj)

        err_g_con_s, err_g_con_x = self.NetGe.loss_func(adj_label, s_fake, h0, x_fake)

        node_loss = torch.mean(F.mse_loss(x1_r, x2, reduction='none'), dim=2).mean(dim=1).mean(dim=0)
        graph_loss = F.mse_loss(Feat_0, Feat_1, reduction='none').mean(dim=1).mean(dim=0)
        err_g_enc = self.NetGe.loss_cal(Feat_0_1, Feat_0)

        lossG = err_g_con_s + err_g_con_x + node_loss + graph_loss + err_g_enc

        return lossG
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
