import time
from inspect import signature
from abc import ABC, abstractmethod

import torch
import numpy as np
from scipy.stats import binom
from scipy.special import erf

from torch_geometric.nn import GIN
from torch_geometric import compile
from torch_geometric.loader import NeighborLoader

# from ..utils import logger, validate_device, pprint, is_fitted


class Detector(ABC):
    

    def __init__(self,
                 contamination=0.1,
                 verbose=0):

        if not (0. < contamination <= 0.5):
            raise ValueError("contamination must be in (0, 0.5], "
                             "got: %f" % contamination)

        self.contamination = contamination
        self.verbose = verbose
        self.decision_score_ = None

    @abstractmethod
    def process_graph(self, data):
       pass

    @abstractmethod
    def fit(self, data,label=None):
      pass

    @abstractmethod
    def decision_function(self, data, label=None):
        pass
       

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False):
        

        is_fitted(self, ['decision_score_', 'threshold_', 'label_'])

        output = ()
        if data is None:
            score = self.decision_score_
            logger(score=self.decision_score_,
                   target=label,
                   verbose=self.verbose,
                   train=False)
        else:
            score = self.decision_function(data, label)
        if return_pred:
            pred = (score > self.threshold_).long()
            output += (pred,)
        if return_score:
            output += (score,)
        if return_prob:
            prob = self._predict_prob(score, prob_method)
            output += (prob,)
        if return_conf:
            conf = self._predict_conf(score)
            output += (conf,)

        if len(output) == 1:
            return output[0]
        else:
            return output

    def _predict_prob(self, score, method='linear'):
        
        if method == 'linear':
            train_score = self.decision_score_
            prob = score - train_score.min()
            prob /= train_score.max() - train_score.min()
            prob = prob.clamp(0, 1)
        elif method == 'unify':
            mu = torch.mean(self.decision_score_)
            sigma = torch.std(self.decision_score_)
            pre_erf_score = (score - mu) / (sigma * np.sqrt(2))
            erf_score = erf(pre_erf_score)
            prob = erf_score.clamp(0, 1)
        else:
            raise ValueError(method,
                             'is not a valid probability conversion method')
        return prob

    def _predict_conf(self, score):
       

        n = len(self.decision_score_)
        k = n - int(n * self.contamination)

        n_ins = (self.decision_score_.view(n, 1) <= score).count_nonzero(dim=0)

        post_prob = (1 + n_ins) / (2 + n)

        conf = torch.Tensor(1 - binom.cdf(k, n, post_prob))

        pred = (score > self.threshold_).long()
        conf = torch.where(pred == 0, 1 - conf, conf)
        return conf

    def _process_decision_score(self):
       

        self.threshold_ = np.percentile(self.decision_score_,
                                        100 * (1 - self.contamination))
        self.label_ = (self.decision_score_ > self.threshold_).long()

    def __repr__(self):

        class_name = self.__class__.__name__
        init_signature = signature(self.__init__)
        parameters = [p for p in init_signature.parameters.values()
                      if p.name != 'self' and p.kind != p.VAR_KEYWORD]
        params = {}
        for key in sorted([p.name for p in parameters]):
            params[key] = getattr(self, key, None)
        return '%s(%s)' % (class_name, pprint(params, offset=len(class_name)))


class DeepDetector(Detector, ABC):
   

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GIN,
                 contamination=0.1,
                 lr=4e-3,
                 epoch=100,
                 gpu=-1,
                 batch_size=0,
                 num_neigh=-1,
                 verbose=0,
                 gan=False,
                 save_emb=False,
                 compile_model=False,
                 **kwargs):

        super(DeepDetector, self).__init__(contamination=contamination,
                                           verbose=verbose)

        # model param
        self.in_dim = in_dim
        self.num_nodes = None
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.str_dim = str_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.act = act
        self.backbone = backbone
        self.kwargs = kwargs

        # training param
        self.lr = lr
        self.epoch = epoch
        self.device = gpu
        self.batch_size = batch_size
        self.gan = gan
        if type(num_neigh) is int:
            self.num_neigh = [num_neigh] * self.num_layers
        elif type(num_neigh) is list:
            if len(num_neigh) != self.num_layers:
                raise ValueError('Number of neighbors should have the '
                                 'same length as hidden layers dimension or'
                                 'the number of layers.')
            self.num_neigh = num_neigh
        else:
            raise ValueError('Number of neighbors must be int or list of int')

        # other param
        self.model = None
        self.save_emb = save_emb
        if save_emb:
            self.emb = None
        self.compile_model = compile_model

    def fit(self, data, label=None):

        self.process_graph(data)
        self.num_nodes, self.in_dim = data.x.shape
        if self.batch_size == 0:
            self.batch_size = data.x.shape[0]

        dataloader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model = self.init_model(**self.kwargs)
        if self.compile_model:
            self.model = compile(self.model)
        if not self.gan:
            optimizer = torch.optim.Adam(self.model.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)
        else:
            self.opt_in = torch.optim.Adam(self.model.inner.parameters(),
                                           lr=self.lr,
                                           weight_decay=self.weight_decay)
            optimizer = torch.optim.Adam(self.model.outer.parameters(),
                                         lr=self.lr,
                                         weight_decay=self.weight_decay)

        self.model.train()
        self.decision_score_ = torch.zeros(data.x.shape[0])
        for epoch in range(self.epoch):
            start_time = time.time()
            epoch_loss = 0
            if self.gan:
                self.epoch_loss_in = 0

            for sampled_data in dataloader:
                batch_size = sampled_data.batch_size
                node_idx = sampled_data.n_id
                if self.model.name == 'GOOD_D':
                    loss, score = self.forward_model(sampled_data,dataloader)
                epoch_loss += loss.item() * batch_size
                if self.save_emb:
                    if type(self.emb) is tuple:
                        self.emb[0][node_idx[:batch_size]] = \
                            self.model.emb[0][:batch_size].cpu()
                        self.emb[1][node_idx[:batch_size]] = \
                            self.model.emb[1][:batch_size].cpu()
                    else:
                        self.emb[node_idx[:batch_size]] = \
                            self.model.emb[:batch_size].cpu()
                self.decision_score_[node_idx[:batch_size]] = score

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_value = epoch_loss / data.x.shape[0]
            if self.gan:
                loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)
            logger(epoch=epoch,
                   loss=loss_value,
                   score=self.decision_score_,
                   target=label,
                   time=time.time() - start_time,
                   verbose=self.verbose,
                   train=True)

        self._process_decision_score()
        return self

    def decision_function(self, data, label=None):

        self.process_graph(data)
        loader = NeighborLoader(data,
                                self.num_neigh,
                                batch_size=self.batch_size)

        self.model.eval()
        outlier_score = torch.zeros(data.x.shape[0])
        if self.save_emb:
            if type(self.hid_dim) is tuple:
                self.emb = (torch.zeros(data.x.shape[0], self.hid_dim[0]),
                            torch.zeros(data.x.shape[0], self.hid_dim[1]))
            else:
                self.emb = torch.zeros(data.x.shape[0], self.hid_dim)
        start_time = time.time()
        test_loss = 0
        for sampled_data in loader:
            loss, score = self.forward_model(sampled_data,loader)
            batch_size = sampled_data.batch_size
            node_idx = sampled_data.n_id
            if self.save_emb:
                if type(self.hid_dim) is tuple:
                    self.emb[0][node_idx[:batch_size]] = \
                        self.model.emb[0][:batch_size].cpu()
                    self.emb[1][node_idx[:batch_size]] = \
                        self.model.emb[1][:batch_size].cpu()
                else:
                    self.emb[node_idx[:batch_size]] = \
                        self.model.emb[:batch_size].cpu()

            test_loss = loss.item() * batch_size
            outlier_score[node_idx[:batch_size]] = score

        loss_value = test_loss / data.x.shape[0]
        if self.gan:
            loss_value = (self.epoch_loss_in / data.x.shape[0], loss_value)

        logger(loss=loss_value,
               score=outlier_score,
               target=label,
               time=time.time() - start_time,
               verbose=self.verbose,
               train=False)
        return outlier_score

    def predict(self,
                data=None,
                label=None,
                return_pred=True,
                return_score=False,
                return_prob=False,
                prob_method='linear',
                return_conf=False,
                return_emb=False):
      
        if return_emb:
            self.save_emb = True

        output = super(DeepDetector, self).predict(data,
                                                   label,
                                                   return_pred,
                                                   return_score,
                                                   return_prob,
                                                   prob_method,
                                                   return_conf)
        if return_emb:
            if type(output) is tuple:
                output += (self.emb,)
            else:
                output = (output, self.emb)

        return output

    @abstractmethod
    def init_model(self, **kwargs):
       pass

    @abstractmethod
    def forward_model(self, data):
        pass
       
