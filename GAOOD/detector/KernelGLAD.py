# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
import time

from .mybase import DeepDetector
import os
from sklearn.ensemble import IsolationForest
import torch, numpy as np
from grakel.kernels import WeisfeilerLehman, VertexHistogram, Propagation, ShortestPath, PropagationAttr
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
class KernelGLAD(DeepDetector):
    def __init__(self, kernel, detector, labeled=True,
                       WL_iter=5, PK_bin_width=1,
                       IF_n_trees=200, IF_sample_ratio=0.5,
                       LOF_n_neighbors=20, LOF_n_leaf=30,
                       detectorskernel='precomputed',
                       nu=0.1,
                       args=None, **kwargs):
        super(KernelGLAD, self).__init__(in_dim=None)
        kernels = {
            'WL': WeisfeilerLehman(n_iter=WL_iter, normalize=True, base_graph_kernel=VertexHistogram),
            'PK': Propagation(t_max=WL_iter, w=PK_bin_width, normalize=True) if labeled else
                    PropagationAttr(t_max=WL_iter, w=PK_bin_width, normalize=True),
        }
        detectors = {
            'OCSVM': OneClassSVM(kernel=detectorskernel, nu=nu),
            'LOF': LocalOutlierFactor(n_neighbors=LOF_n_neighbors, leaf_size=LOF_n_leaf,
                                      metric='precomputed', contamination=0.1),
            'IF': IsolationForest(n_estimators=IF_n_trees, max_samples=IF_sample_ratio, contamination='auto'),
        }

        assert kernel in kernels.keys()
        assert detector in detectors.keys()

        self.kernel = kernels[kernel]
        self.detector = detectors[detector]
        self.kernel_name = kernel
        self.detector_name = detector
        self.labeled = labeled
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
        pass

    def to_grakel_dataset(self, dataset, labeled=True):
        def to_grakel_graph(data, labeled=True):
            edges = {tuple(edge) for edge in data.edge_index.T.numpy()}
            if labeled:
                labels = {i: fea.argmax().item() + 1 for i, fea in enumerate(data.x)}
            else:
                labels = {i: fea.numpy() for i, fea in enumerate(data.x)}
            return [edges, labels]

        return [to_grakel_graph(data, labeled) for data in dataset]

    def _calculate_kernel_matrix_pk(self, kernel, normalize=True):
        def pairwise_operation(x, y, kernel):
            return np.array([kernel.metric(x[t], y[t]) for t in range(kernel.t_max)])

        X = kernel.X
        kernel_matrices = np.zeros(shape=(kernel.t_max, len(X), len(X)))
        cache = list()
        for (i, x) in enumerate(X):
            kernel_matrices[:, i, i] = pairwise_operation(x, x, kernel)
            for (j, y) in enumerate(cache):
                kernel_matrices[:, j, i] = pairwise_operation(y, x, kernel)
            cache.append(x)
        for i in range(kernel.t_max):
            kernel_matrices[i] = np.triu(kernel_matrices[i]) + np.triu(kernel_matrices[i], 1).T

        accumulative_kernel_matrices = np.add.accumulate(kernel_matrices, 0)

        if normalize:
            for i in range(kernel.t_max):
                _X_diag = np.diagonal(kernel_matrices[i])
                kernel_matrices[i] = kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))

                _X_diag = np.diagonal(accumulative_kernel_matrices[i])
                accumulative_kernel_matrices[i] = accumulative_kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))

        return kernel_matrices, accumulative_kernel_matrices

    def _calculate_kernel_matrix_wl(self, kernel, normalize=True):
        base_kernels = kernel.X  # length = wl-iteration
        n_wl_iters = len(base_kernels)
        kernel_matrices = np.stack([base_kernels[i]._calculate_kernel_matrix() for i in range(n_wl_iters)],
                                   axis=0).astype(float)  # unormalized
        accumulative_kernel_matrices = np.add.accumulate(kernel_matrices, 0)

        if normalize:
            for i in range(n_wl_iters):
                _X_diag = np.diagonal(kernel_matrices[i]) + 1e-6
                kernel_matrices[i] = kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))

                _X_diag = np.diagonal(accumulative_kernel_matrices[i]) + 1e-6
                accumulative_kernel_matrices[i] = accumulative_kernel_matrices[i] / np.sqrt(np.outer(_X_diag, _X_diag))
        return kernel_matrices, accumulative_kernel_matrices

    def init_model(self, **kwargs):
        '''
        :param embedders, detectors:
        :return: Graph2Vec, FGSD, IF, LOF, OCSVM
        '''
        return (self.kernel,self.detector)


    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        dataset = self.to_grakel_dataset(dataloader_val, self.labeled)

        self.kernel_matrix = self.kernel.fit_transform(dataset)
        if self.detector_name in ['IF', 'OCSVM'] :
            self.detector.fit(self.kernel_matrix)
            self.anomaly_scores = -self.detector.decision_function(self.kernel_matrix)
        else:
            self.detector.fit(np.amax(self.kernel_matrix) - self.kernel_matrix)
            self.anomaly_scores = -self.detector.negative_outlier_factor_
        # return True
    def is_directory_empty(self,directory):
        # 列出目录下的所有文件和文件夹
        files_and_dirs = os.listdir(directory)
        # 如果列表为空，则目录为空
        return len(files_and_dirs) == 0

    def decision_function(self, dataset, label=None, dataloader=None, args=None):

        ys = torch.cat([data.y for data in dataloader])
        return self.anomaly_scores, ys


    def forward_model(self, dataloader):
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