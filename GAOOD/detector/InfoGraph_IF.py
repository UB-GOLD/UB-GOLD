# -*- coding: utf-8 -*-
from .mybase import DeepDetector

from ..nn import Infograph
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import torch
import os
from GAOOD.metric import *
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
import torch.nn.functional as F

from torch import nn
from tqdm import tqdm
from torch.optim import Adam
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from GCL.models import SingleBranchContrast
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset
from sklearn.svm import OneClassSVM
import joblib
class InfoGraph_IF(DeepDetector):
    def __init__(self,
                 DS='BZR',
                 DS_pair=None,
                 exp_type=None,
                 model_name=None,
                 args=None,
                 detector='IF',
                 gamma='scale',
                 nu=0.1,
                 IF_n_trees=200, IF_sample_ratio=0.5,
                 **kwargs):
        super(InfoGraph_IF, self).__init__(in_dim=None)
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
        :return: InfoGraph_IF
        '''
        self.gconv = Infograph.GConv(input_dim=self.args.dataset_num_features,
                      hidden_dim=self.args.hidden_dim, activation=torch.nn.ReLU,
                    num_layers=self.args.num_layer).to(self.device)
        self.fc1 = Infograph.FC(hidden_dim=self.args.hidden_dim * self.args.num_layer)
        self.fc2 = Infograph.FC(hidden_dim=self.args.hidden_dim * self.args.num_layer)
        self.encoder_model = Infograph.Encoder(encoder=self.gconv, local_fc=self.fc1, global_fc=self.fc2).to(self.device)
        self.contrast_model = SingleBranchContrast(loss=L.JSD(), mode='G2L').to(self.device)


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
                z, g = self.encoder_model(data.x, data.edge_index, data.batch)
                z, g = self.encoder_model.project(z, g)
                self.detector.fit(g.cpu().detach().numpy())
                loss = self.contrast_model(h=z, g=g, batch=data.batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.encoder_model.eval()
                ys = torch.cat([data.y for data in dataloader_val])
                y_score_all = []
                for data in dataloader_val:
                    data = data.to(self.device)
                    if data.x is None:
                        num_nodes = data.batch.size(0)
                        data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
                    z, g = self.encoder_model(data.x, data.edge_index, data.batch)
                    # self.detector.fit(g.cpu().detach().numpy())
                    anomaly_scores = -self.detector.decision_function(g.cpu().detach().numpy())
                    y_score_all = y_score_all + list(anomaly_scores)
                val_auc = ood_auc(ys, y_score_all)
                if val_auc > max_AUC:
                    max_AUC = val_auc
                    torch.save(self.encoder_model, os.path.join(self.path, 'Infograph_encoder_model.pth'))
                    joblib.dump(self.detector, os.path.join(self.path, 'Infograph_{}_model.joblib'.format(self.detector_name)))


        return True
    def is_directory_empty(self,directory):
        files_and_dirs = os.listdir(directory)
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        if self.is_directory_empty(self.path):
            pass
        else:
            self.encoder_model = torch.load(os.path.join(self.path, 'Infograph_encoder_model.pth'))
            self.detector = joblib.load(os.path.join(self.path, 'Infograph_{}_model.joblib'.format(self.detector_name)))
        self.encoder_model.eval()
        self.device = torch.device('cuda:' + str(self.args.gpu) if torch.cuda.is_available() else 'cpu')
        ys = torch.cat([data.y for data in dataloader])
        y_score_all = []
        for data in dataloader:
            data = data.to(self.device)
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            z, g = self.encoder_model(data.x, data.edge_index, data.batch)
            anomaly_scores = -self.detector.decision_function(g.cpu().detach().numpy())
            y_score_all = y_score_all + list(anomaly_scores)

        return y_score_all, ys


    def forward_model(self, dataset, dataloader=None, args=None):
        pass
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
