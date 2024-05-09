# -*- coding: utf-8 -*-
""" One-Class Graph Neural Networks for Anomaly Detection in Attributed
Networks"""
# Author: Xueying Ding <xding2@andrew.cmu.edu>
# License: BSD 2 clause

import torch
from torch_geometric.nn import GCN

from .mybase import DeepDetector
from ..nn import graphde
import faiss
import numpy as np
from GAOOD.metric import *
import os


def run_kmeans(x, args):
    results = {}

    d = x.shape[1]
    k = args.num_cluster
    clus = faiss.Clustering(d, k)
    clus.niter = 20
    clus.nredo = 5
    clus.seed = 0
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 3

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False

    try:
        index = faiss.GpuIndexFlatL2(res, d, cfg)
        clus.train(x, index)
    except:
        print('Fail to cluster with GPU. Try CPU...')
        index = faiss.IndexFlatL2(d)
        clus.train(x, index)

    D, I = index.search(x, 1)
    im2cluster = [int(n[0]) for n in I]

    centroids = faiss.vector_to_array(clus.centroids).reshape(k, d)

    Dcluster = [[] for c in range(k)]
    for im, i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])

    density = np.zeros(k)
    for i, dist in enumerate(Dcluster):
        if len(dist) > 1:
            d = (np.asarray(dist) ** 0.5).mean() / np.log(len(dist) + 10)
            density[i] = d

    dmax = density.max()
    for i, dist in enumerate(Dcluster):
        if len(dist) <= 1:
            density[i] = dmax

    density = density.clip(np.percentile(density, 30),
                           np.percentile(density, 70))
    density = density / density.mean() + 0.5

    centroids = torch.Tensor(centroids).cuda()
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

    im2cluster = torch.LongTensor(im2cluster).cuda()
    density = torch.Tensor(density).cuda()

    results['centroids'] = centroids
    results['density'] = density
    results['im2cluster'] = im2cluster

    return results


class GraphDE(DeepDetector):
    """
    One-Class Graph Neural Networks for Anomaly Detection in
    Attributed Networks

    OCGNN is an anomaly detector that measures the
    distance of anomaly to the centroid, in a similar fashion to the
    support vector machine, but in the embedding space after feeding
    towards several layers of GCN.

    See :cite:`wang2021one` for details.

    Parameters
    ----------
    hid_dim :  int, optional
        Hidden dimension of model. Default: ``64``.
    num_layers : int, optional
        Total number of layers in model. Default: ``2``.
    dropout : float, optional
        Dropout rate. Default: ``0.``.
    weight_decay : float, optional
        Weight decay (L2 penalty). Default: ``0.``.
    act : callable activation function or None, optional
        Activation function if not None.
        Default: ``torch.nn.functional.relu``.
    backbone : torch.nn.Module
        The backbone of the deep detector implemented in PyG.
        Default: ``torch_geometric.nn.GCN``.
    contamination : float, optional
        The amount of contamination of the dataset in (0., 0.5], i.e.,
        the proportion of outliers in the dataset. Used when fitting to
        define the threshold on the decision function. Default: ``0.1``.
    lr : float, optional
        Learning rate. Default: ``0.004``.
    epoch : int, optional
        Maximum number of training epoch. Default: ``100``.
    gpu : int
        GPU Index, -1 for using CPU. Default: ``-1``.
    batch_size : int, optional
        Minibatch size, 0 for full batch training. Default: ``0``.
    num_neigh : int, optional
        Number of neighbors in sampling, -1 for all neighbors.
        Default: ``-1``.
    beta : float, optional
        The weight between the reconstruction loss and radius.
        Default: ``0.5``.
    warmup : int, optional
        The number of epochs for warm-up training. Default: ``2``.
    eps : float, optional
        The slack variable. Default: ``0.001``.
    verbose : int, optional
        Verbosity mode. Range in [0, 3]. Larger value for printing out
        more log information. Default: ``0``.
    save_emb : bool, optional
        Whether to save the embedding. Default: ``False``.
    compile_model : bool, optional
        Whether to compile the model with ``torch_geometric.compile``.
        Default: ``False``.
    **kwargs
        Other parameters for the backbone model.

    Attributes
    ----------
    decision_score_ : torch.Tensor
        The outlier scores of the training data. Outliers tend to have
        higher scores. This value is available once the detector is
        fitted.
    threshold_ : float
        The threshold is based on ``contamination``. It is the
        :math:`N \\times` ``contamination`` most abnormal samples in
        ``decision_score_``. The threshold is calculated for generating
        binary outlier labels.
    label_ : torch.Tensor
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers. It is generated by applying
        ``threshold_`` on ``decision_score_``.
    emb : torch.Tensor or tuple of torch.Tensor or None
        The learned node hidden embeddings of shape
        :math:`N \\times` ``hid_dim``. Only available when ``save_emb``
        is ``True``. When the detector has not been fitted, ``emb`` is
        ``None``. When the detector has multiple embeddings,
        ``emb`` is a tuple of torch.Tensor.
    """

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

    def fit(self, dataset, args=None, label=None, dataloader=None, dataloader_Val=None):

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
            if (epoch) % 5 == 0 and epoch > 0:
                self.model.eval()

                y_val = []
                score_val = []
                for data in dataloader_Val:
                    data = data.to(device)
                    score_epoch = self.model.infer_e_gx(data.x, data.edge_index, data.batch)
                    score_val = score_val + score_epoch.detach().cpu().tolist()
                    y_true = data.y
                    y_val = y_val + y_true.detach().cpu().tolist()

                val_auc = ood_auc(y_val, score_val)

                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    torch.save(self.model, os.path.join(self.path, 'model_GOOD_D.pth'))
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
            self.model = torch.load(os.path.join(self.path, 'model_GOOD_D.pth'))
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
            score, y_all = self.decision_function(dataset, label, dataloader, args)
            output = (score, y_all)
            return output