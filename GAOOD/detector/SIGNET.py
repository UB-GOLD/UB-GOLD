# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
import torch_scatter
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, HypergraphConv, global_add_pool, global_max_pool
from torch_geometric.utils import softmax
from .mybase import DeepDetector
from ..nn import signet
import os
from GAOOD.metric import *
class SIGNET(DeepDetector):
    def __init__(self,
                 DS='BZR',
                 DS_pair=None,
                 exp_type=None,
                 model_name=None,
                 input_dim=16,
                 input_dim_edge=16,
                 args=None,
                 **kwargs):
        super(SIGNET, self).__init__(in_dim=None)
        self.DS = DS
        self.DS_pair = DS_pair
        self.exp_type = exp_type
        self.model_name = model_name
        self.input_dim = input_dim
        self.input_dim_edge = input_dim_edge
        self.args = args
        self.build_save_path()

    def build_save_path(self):
        path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.exp_type == 'oodd':
            path = os.path.join(path, 'model_save', self.model_name, self.exp_type, self.DS_pair)
        elif self.DS.startswith('Tox21'):
            path = os.path.join(path, 'model_save', self.model_name, self.exp_type + 'Tox21', self.DS)
        else:
            path = os.path.join(path, 'model_save', self.model_name, self.exp_type, self.DS)
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
        :return: SIGNET
        '''
        return signet.SIGNET(input_dim=self.input_dim,
                            input_dim_edge=self.input_dim_edge,
                            args=self.args,
                            device=self.device,
                            **kwargs).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        max_AUC=0
        self.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        for epoch in range(1, args.num_epoch + 1):
            self.model.train()
            loss_all = 0
            num_sample = 0
            for data in dataloader:
                optimizer.zero_grad()
                data = data.to(self.device)
                loss = self.forward_model(dataset = data)
                loss_all += loss.item() * data.num_graphs
                num_sample += data.num_graphs
                loss.backward()
                optimizer.step()
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model.eval()
                # anomaly detection
                all_ad_true = []
                all_ad_score = []
                for data in dataloader_val:
                    all_ad_true.append(data.y.cpu())
                    data = data.to(self.device)
                    with torch.no_grad():
                        y, y_hyper, _, _ = self.model(data)
                        ano_score = self.model.loss_nce(y, y_hyper)
                    all_ad_score.append(ano_score.cpu())

                ad_true = torch.cat(all_ad_true)
                ad_score = torch.cat(all_ad_score)
                val_auc = ood_auc(ad_true, ad_score)
                if val_auc > max_AUC:
                    max_AUC = val_auc
                    torch.save(self.model, os.path.join(self.path, 'model_SIGNET.pth'))
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
            self.model = torch.load(os.path.join(self.path, 'model_SIGNET.pth'))
        self.model.eval()
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
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

        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score,y_all = self.decision_function(dataset, label,dataloader,args)
            output = (score,y_all)
            return output
