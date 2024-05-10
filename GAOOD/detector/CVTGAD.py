# -*- coding: utf-8 -*-
from .mybase import DeepDetector
from ..nn import cvtgad
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import numpy as np
import torch
import os
from GAOOD.metric import *
class CVTGAD(DeepDetector):
    def __init__(self,
                 DS='BZR',
                 DS_pair=None,
                 exp_type=None,
                 model_name=None,
                 args=None,
                 **kwargs):
        super(CVTGAD, self).__init__(in_dim=None)
        self.DS = DS
        self.DS_pair = DS_pair
        self.exp_type = exp_type
        self.model_name = model_name
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
        :return: CVTGAD
        '''
        return cvtgad.CVTGAD(hidden_dim = self.args.hidden_dim,
                          num_gc_layers = self.args.num_layer,
                          feat_dim = self.args.dataset_num_features,
                          str_dim = self.args.dg_dim + self.args.rw_dim,
                          args = self.args).to(self.device)

    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.max_AUC = 0
        stop_counter = 0  # 初始化停止计数器
        N = 5  # 设定阈值，比如连续5次AUC没有提升就停止

        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_g, weight_n = 1, 1
                else:
                    weight_g, weight_n = std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_g  + weight_n) / 2
                    weight_g, weight_n = weight_g/weight_sum, weight_n/weight_sum
            # cluster_result = get_cluster_result(dataloader, model, args)
            self.model.train()
            loss_all = 0
            if args.is_adaptive:
                loss_g_all, loss_n_all = [], []
            for data in dataloader:
                data = data.to(self.device)
                optimizer.zero_grad()
                loss_g, loss_n = self.forward_model(data, dataloader, args)
                if args.is_adaptive:
                    loss = weight_g * loss_g.mean() + weight_n * loss_n.mean()
                    # loss_b_all = loss_b_all + loss_b.detach().cpu().tolist()
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                else:
                    loss = loss_g.mean() + loss_n.mean()
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
            if args.is_adaptive:
                # mean_b, std_b = np.mean(loss_b_all), np.std(loss_b_all)
                mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
            if (epoch) % args.eval_freq == 0 and epoch > 0:

                self.model.eval()
                y_score_all = []
                y_true_all = []

                for data in dataloader_val:
                    data = data.to(self.device)
                    y_score_g, y_score_n = self.forward_model(data, dataloader_val, args)
                    if args.is_adaptive:
                        y_score = (y_score_g - mean_g) / std_g + (y_score_n - mean_n) / std_n
                    else:
                        y_score = y_score_g + y_score_n
                    y_true = data.y

                    y_score_all = y_score_all + y_score.detach().cpu().tolist()
                    y_true_all = y_true_all + y_true.detach().cpu().tolist()

                val_auc = ood_auc(y_true_all, y_score_all)
                
                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    stop_counter = 0  # 重置计数器
                    torch.save(self.model, os.path.join(self.path, 'model_CVTGAD.pth'))
                else:
                    stop_counter += 1  # 增加计数器
                print('[TRAIN] Epoch:{:03d} | val_auc:{:.4f}'.format(epoch, self.max_AUC))
                if stop_counter >= N:
                    print(f'Early stopping triggered after {epoch} epochs due to no improvement in AUC for {N} consecutive evaluations.')
                    break  # 达到早停条件，跳出循环
                
        return True
    def is_directory_empty(self,directory):
        # 列出目录下的所有文件和文件夹
        files_and_dirs = os.listdir(directory)
        # 如果列表为空，则目录为空
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        if self.is_directory_empty(self.path):
            print("Can't find the path")
        else:
            print("Loading Model Weight")
            self.model = torch.load(os.path.join(self.path, 'model_CVTGAD.pth'))
        self.model.eval()
        self.device = torch.device('cuda:' + str(self.args.gpu) if torch.cuda.is_available() else 'cpu')
        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(self.device)
            y_score_g, y_score_n = self.forward_model(data, dataloader, args)
            y_score = y_score_g + y_score_n
            y_true = data.y
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true_all = y_true_all + y_true.detach().cpu().tolist()

        return y_score_all, y_true_all


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


        output = ()
        if dataset is None:
            score = self.decision_score_

        else:
            score,y_all = self.decision_function(dataset, label,dataloader,args)
            output = (score,y_all)
            return output
