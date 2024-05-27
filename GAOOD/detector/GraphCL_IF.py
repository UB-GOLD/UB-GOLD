# -*- coding: utf-8 -*-
from .mybase import DeepDetector
from ..nn import graphcl
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import torch
import os
from GAOOD.metric import *
import torch
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from GCL.eval import get_split, SVMEvaluator
from GCL.models import DualBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.svm import OneClassSVM
import joblib
class GraphCL_IF(DeepDetector):
    def __init__(self,
                 DS='BZR',
                 DS_pair=None,
                 exp_type=None,
                 model_name=None,
                 args=None,
                 detector='IF',
                 gamma ='scale',
                 nu=0.1,
                 IF_n_trees=200, IF_sample_ratio=0.5,
                 **kwargs):
        super(GraphCL_IF, self).__init__(in_dim=None)
        detectors = {
            'IF': IsolationForest(n_estimators=IF_n_trees, max_samples=IF_sample_ratio, contamination='auto'),
            'OCSVM': OneClassSVM(gamma=gamma, nu=nu)
        }
        self.DS = DS
        self.DS_pair = DS_pair
        self.exp_type = exp_type
        self.model_name = model_name
        self.args = args
        self.detector = detectors[detector]
        self.detector_name = detector
        self.build_save_path()

    def build_save_path(self):
        print(self.args)
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.args.exp_type == 'oodd':
            path = os.path.join(path, 'model_save', self.args.model, self.args.exp_type, self.args.DS_pair)
        elif self.args.DS.startswith('Tox21'):
            path = os.path.join(path, 'model_save', self.args.model, self.args.exp_type + 'Tox21', self.args.DS)
        else:
            path = os.path.join(path, 'model_save', self.args.model, self.args.exp_type, self.args.DS)
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
        pass

    def init_model(self, **kwargs):
        '''
        :param kwargs:
        :return: CVTGAD
        '''
        aug1 = A.Identity()
        aug2 = A.RandomChoice([A.RWSampling(num_seeds=1000, walk_length=10),
                               A.NodeDropping(pn=0.1),
                               A.FeatureMasking(pf=0.1),
                               A.EdgeRemoving(pe=0.1)], 1)
        self.gconv = graphcl.GConv(input_dim=self.args.dataset_num_features,
                      hidden_dim=self.args.hidden_dim, num_layers=self.args.num_layer
                      ).to(self.device)
        self.encoder_model =  graphcl.Encoder(encoder=self.gconv, augmentor=(aug1, aug2)).to(self.device)
        self.contrast_model = DualBranchContrast(loss=L.InfoNCE(tau=0.2), mode='G2G').to(self.device)

        # self.detector = IsolationForest(n_estimators=200, max_samples=0.5, contamination=0.1)
        return True

    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):

        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.init_model(**self.kwargs)
        optimizer = Adam(self.encoder_model.parameters(), lr=args.lr)
        max_AUC = 0
        for epoch in range(1, args.num_epoch + 1):
            self.encoder_model.train()
            epoch_loss = 0
            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                if data.x is None:
                    num_nodes = data.batch.size(0)
                    data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)

                _, g, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
                self.detector.fit(g.cpu().detach().numpy())
                g1, g2 = [self.encoder_model.encoder.project(g) for g in [g1, g2]]
                loss = self.contrast_model(g1=g1, g2=g2, batch=data.batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            # print(epoch_loss)
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.encoder_model.eval()
                ys = torch.cat([data.y for data in dataloader_val])
                y_score_all = []
                for data in dataloader_val:
                    data = data.to(self.device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                    _, g, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
                    # print(g)
                    # self.detector.fit(g.cpu().detach().numpy())
                    anomaly_scores = -self.detector.decision_function(g.cpu().detach().numpy())
                    y_score_all = y_score_all + list(anomaly_scores)
                val_auc = ood_auc(ys, y_score_all)
                if val_auc > max_AUC:
                    print("保存模型： ",val_auc)
                    max_AUC = val_auc
                    torch.save(self.encoder_model, os.path.join(self.path, 'encoder_model.pth'))
                    joblib.dump(self.detector, os.path.join(self.path, 'isolation_forest_model.joblib'))
        return True
    def is_directory_empty(self,directory):
        # 列出目录下的所有文件和文件夹
        files_and_dirs = os.listdir(directory)
        # 如果列表为空，则目录为空
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        if self.is_directory_empty(self.path):
            pass
        else:
            print("加载模型： ")
            self.encoder_model = torch.load(os.path.join(self.path, 'encoder_model.pth'))
            self.detector = joblib.load(os.path.join(self.path, 'isolation_forest_model.joblib'))
        self.encoder_model.eval()
        self.device = torch.device('cuda:' + str(self.args.gpu) if torch.cuda.is_available() else 'cpu')
        ys = torch.cat([data.y for data in dataloader])
        y_score_all = []
        for data in dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, g, _, _, g1, g2 = self.encoder_model(data.x, data.edge_index, data.batch)
            anomaly_scores = -self.detector.decision_function(g.cpu().detach().numpy())
            y_score_all = y_score_all + list(anomaly_scores)

        return y_score_all, ys


    def forward_model(self, dataset, dataloader=None, args=None):
        g_f, g_s, n_f, n_s = self.model(dataset.x, dataset.x_s, dataset.edge_index, dataset.batch, dataset.num_graphs)
        loss_g = self.model.calc_loss_g(g_f, g_s)
        loss_n = self.model.calc_loss_n(n_f, n_s, dataset.batch)

        return loss_g, loss_n
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