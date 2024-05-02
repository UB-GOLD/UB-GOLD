# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax
from .mybase import DeepDetector
from ..nn import signet

class SIGNET(DeepDetector):
    def __init__(self,
                 num_epochs=100,
                 gpu=0,
                 lr=0.01,
                 input_dim=16,
                 input_dim_edge=16,
                 args=None,
                 **kwargs):
        super(SIGNET, self).__init__(in_dim=None)

        self.num_epochs = num_epochs
        self.lr = lr
        self.gpu = gpu
        self.input_dim = input_dim
        self.input_dim_edge = input_dim_edge
        self.args = args

    def process_graph(self, data):
        pass

    def init_model(self, **kwargs):
        '''
        :param kwargs:
        :return: SIGNET
        '''
        return signet.SIGNET(input_dim=self.input_dim,
                            input_dim_edge=self.input_dim_edge,
                            args=self.args,
                            device=self.device,
                            **kwargs).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None):

        self.device = torch.device('cuda:'+str(self.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.train_dataloader = dataloader
        for epoch in range(1, self.num_epochs + 1):
            self.model.train()
            loss_all = 0
            num_sample = 0
            for data in self.train_dataloader:
                optimizer.zero_grad()
                data = data.to(self.device)
                loss = self.forward_model(dataset = data)
                loss_all += loss.item() * data.num_graphs
                num_sample += data.num_graphs
                loss.backward()
                optimizer.step()
        return True


    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        self.model.eval()
        self.device = torch.device('cuda:' + str(self.gpu) if torch.cuda.is_available() else 'cpu')
        all_ad_true = []
        all_ad_score = []
        for data in dataloader:
            all_ad_true.append(data.y.cpu())
            data = data.to(self.device)
            with torch.no_grad():
                y, y_hyper, _, _ = self.model(data)
                ano_score = self.model.loss_nce(y, y_hyper)
            all_ad_score.append(ano_score.cpu())
        ad_true = torch.cat(all_ad_true)
        ad_score = torch.cat(all_ad_score)
        return ad_score, ad_true


    def forward_model(self, dataset, dataloader=None, args=None):
        y, y_hyper, node_imp, edge_imp = self.model(dataset)
        loss = self.model.loss_nce(y, y_hyper).mean()

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