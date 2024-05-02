# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
import time
from torch.autograd import Variable
import torch.nn.functional as F
from .mybase import DeepDetector
from ..nn import gladc
import scipy.sparse as sp

class GLADC(DeepDetector):
    def __init__(self,
                 max_nodes=0,
                 num_epochs=1,
                 batch_size=300,
                 hidden_dim=256,
                 output_dim=128,
                 num_gc_layers=2,
                 nobn = True,
                 dropout=0.1,
                 lr=0.0001,
                 nobias=True,
                 feature = 'default',
                 seed = 2,
                 device = 0,
                 feature_dim = 53,
                 max_nodes_num = 0,
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
            if name.split('.')[0] == 'proj_head':
                adv_param.data = param.data
            else:
                adv_param.data = param.data + 1.0 * torch.normal(0, torch.ones_like(param.data) * param.data.std()).to(
                    self.device)
        x1_r, Feat_0 = vice_model(h0, adj)
        return x1_r, Feat_0

    def fit(self, dataset, args=None, label=None, dataloader=None):

        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.NetGe, self.noise_NetG = self.init_model(**self.kwargs)
        optimizerG = torch.optim.Adam(self.NetGe.parameters(), lr=self.lr)
        self.train_dataloader = dataloader
        for epoch in range(1, self.num_epochs + 1):
            total_lossG = 0.0
            self.NetGe.train()
            for batch_idx, data in enumerate(self.train_dataloader):
                adj_matrixs, adj_labels, graph_appends, _ = self.process_graph(data)
                lossG = self.forward_model(adj_matrixs, adj_labels, graph_appends)
                optimizerG.zero_grad()
                lossG.backward()
                optimizerG.step()
                total_lossG += lossG
        return True

    def get_adj(self, edge_index, num_nodes):
        adj = torch.zeros(num_nodes, num_nodes)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1
        return adj

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        self.NetGe.eval()
        loss = []
        y = []
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

        for batch_idx, data in enumerate(dataloader):
            adj_matrixs, _, graph_appends,graph_label = self.process_graph(data)
            adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
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

        fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
        test_roc_ab = auc(fpr_ab, tpr_ab)
        print('semi-supervised abnormal detection: auroc_ab: {}'.format(test_roc_ab))
        return label_test,y


    def forward_model(self, adj_matrixs, adj_labels, graph_appends, args=None):
        adj = Variable(adj_matrixs.float(), requires_grad=False).to(self.device)  # .cuda()
        h0 = Variable( graph_appends.float(), requires_grad=False).to(self.device)  # .cuda()
        adj_label = Variable(adj_labels.float(), requires_grad=False).to(self.device)  # .cuda()

        x1_r, Feat_0 = self.NetGe.shared_encoder(h0, adj)
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
        """Prediction for testing data using the fitted detector.
        Return predicted labels by default.

        Parameters
        ----------
        data : torch_geometric.data.Data, optional
            The testing graph. If ``None``, the training data is used.
            Default: ``None``.
        label : torch.Tensor, optional
            The optional outlier ground truth labels used for testing.
            Default: ``None``.
        return_pred : bool, optional
            Whether to return the predicted binary labels. The labels
            are determined by the outlier contamination on the raw
            outlier scores. Default: ``True``.
        return_score : bool, optional
            Whether to return the raw outlier scores.
            Default: ``False``.
        return_prob : bool, optional
            Whether to return the outlier probabilities.
            Default: ``False``.
        prob_method : str, optional
            The method to convert the outlier scores to probabilities.
            Two approaches are possible:

            1. ``'linear'``: simply use min-max conversion to linearly
            transform the outlier scores into the range of
            [0,1]. The model must be fitted first.

            2. ``'unify'``: use unifying scores,
            see :cite:`kriegel2011interpreting`.

            Default: ``'linear'``.
        return_conf : boolean, optional
            Whether to return the model's confidence in making the same
            prediction under slightly different training sets.
            See :cite:`perini2020quantifying`. Default: ``False``.
        return_emb : bool, optional
            Whether to return the learned node representations.
            Default: ``False``.

        Returns
        -------
        pred : torch.Tensor
            The predicted binary outlier labels of shape :math:`N`.
            0 stands for inliers and 1 for outliers.
            Only available when ``return_label=True``.
        score : torch.Tensor
            The raw outlier scores of shape :math:`N`.
            Only available when ``return_score=True``.
        prob : torch.Tensor
            The outlier probabilities of shape :math:`N`.
            Only available when ``return_prob=True``.
        conf : torch.Tensor
            The prediction confidence of shape :math:`N`.
            Only available when ``return_conf=True``.
        """


        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score,y_all = self.decision_function(dataset, label,dataloader,args)
            output = (score,y_all)
            return output