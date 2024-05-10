# -*- coding: utf-8 -*-

import torch
from torch_geometric.nn import GCN

from .mybase import DeepDetector
from ..nn import graphde
import faiss
import numpy as np
from GAOOD.metric import *
import os

class GraphDE(DeepDetector):
  

    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone="GCN",
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 beta=0.5,
                 warmup=2,
                 eps=0.001,
                 verbose=0,
                 save_emb=False,
                 compile_model=False,
                 n_train_data=None,
                 prior_ratio=0,
                 dropedge=0.0,
                 bn=None,
                 dropnode=0.0,
                 neg_ratio=0.1,
                 tem=0.5,
                 sample=4,
                 order=5,
                 lam=1.,
                 grand=False,
                 graphde_a=True,
                 graphde_v=False,
                 args=None,
                 **kwargs):
        super(GraphDE, self).__init__(
            in_dim=in_dim,
            hid_dim=hid_dim,
            num_layers=num_layers,
            str_dim=str_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            act=act,
            backbone=backbone,
            contamination=contamination,
            lr=lr,
            epoch=epoch,
            gpu=gpu,
            batch_size=batch_size,
            num_neigh=num_neigh,
            verbose=verbose,
            save_emb=save_emb,
            compile_model=compile_model,

            **kwargs)

        self.beta = beta
        self.warmup = warmup
        self.eps = eps
        self.n_train_data = n_train_data,
        self.prior_ratio = prior_ratio,
        self.dropedge = dropedge,
        self.bn = bn,
        self.dropnode = dropnode,
        self.neg_ratio = neg_ratio,
        self.tem = tem,
        self.sample = sample,
        self.order = order,
        self.lam = lam,
        self.grand = grand,
        self.graphde_a = graphde_a,
        self.graphde_v = graphde_v,
        self.args = args
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
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)

        return graphde.GraphDE(in_dim=self.in_dim,
                               hid_dim=self.hid_dim,
                               num_layers=self.num_layers,
                               dropout=self.dropout,
                               backbone=self.backbone,
                               n_train_data=self.n_train_data,
                               prior_ratio=self.prior_ratio,
                               dropedge=self.dropedge,
                               bn=self.bn,
                               dropnode=self.dropnode,
                               neg_ratio=self.neg_ratio,
                               tem=self.tem,
                               sample=self.sample,
                               order=self.order,
                               lam=self.lam,
                               grand=self.grand,
                               graphde_a=self.graphde_a,
                               graphde_v=self.graphde_v,
                               **kwargs).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None, dataloader_val=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()
        self.decision_score_ = None
        self.max_AUC = 0
        for epoch in range(1, self.epoch + 1):
            all_loss, n_bw = 0, 0
            for data in dataloader:
                n_bw += 1
                data = data.to(device)
                loss_epoch, score_epoch = self.forward_model(data, dataloader, args)
                all_loss += loss_epoch
            all_loss /= n_bw
            optimizer.zero_grad()
            all_loss.backward()
            # get_gpu_memory_map() # evaluate gpu usage
            optimizer.step()

            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, all_loss))
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model.eval()

                y_val = []
                score_val = []
                for data in dataloader_val:
                    data = data.to(device)
                    score_epoch = self.model.infer_e_gx(data.x, data.edge_index, data.batch)
                    score_val = score_val + score_epoch.detach().cpu().tolist()
                    y_true = data.y
                    y_val = y_val + y_true.detach().cpu().tolist()

                val_auc = ood_auc(y_val, score_val)

                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    torch.save(self.model, os.path.join(self.path, 'GraphDE.pth'))
                print('Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, self.max_AUC))
        return self

    def is_directory_empty(self, directory):
        # 列出目录下的所有文件和文件夹
        files_and_dirs = os.listdir(directory)
        # 如果列表为空，则目录为空
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        if self.is_directory_empty(self.path):
            print("Can't find the path")
        else:
            self.model = torch.load(os.path.join(self.path, 'GraphDE.pth'))
        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(device)
            score_epoch = self.model.infer_e_gx(data.x, data.edge_index, data.batch)
            y_score_all = y_score_all + score_epoch.detach().cpu().tolist()
            y_true = data.y
            y_true_all = y_true_all + y_true.detach().cpu().tolist()

        return y_score_all, y_true_all

    def forward_model(self, data, dataloader=None, args=None):

        loss, score = self.model.loss_func(data)

        return loss, score

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
            score, y_all = self.decision_function(dataset, label, dataloader, args)
            output = (score, y_all)
            return output
