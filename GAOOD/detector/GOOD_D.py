
import torch
from torch_geometric.nn import GCN

from .mybase import DeepDetector
from ..nn import good_d
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


class GOOD_D(DeepDetector):


    def __init__(self,
                 in_dim=None,
                 hid_dim=64,
                 num_layers=2,
                 str_dim=64,
                 dropout=0.,
                 weight_decay=0.,
                 act=torch.nn.functional.relu,
                 backbone=GCN,
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
                 args = None,
                 **kwargs):
        super(GOOD_D, self).__init__(in_dim=in_dim,
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

    def init_model(self, **kwargs):
        if self.save_emb:
            self.emb = torch.zeros(self.num_nodes,
                                   self.hid_dim)

        return good_d.GOOD_D(in_dim=self.in_dim,
                             hid_dim=self.hid_dim,
                             str_dim=self.str_dim,
                             num_layers=self.num_layers,
                             dropout=self.dropout,
                             act=self.act,
                             beta=self.beta,
                             warmup=self.warmup,
                             eps=self.eps,
                             backbone=self.backbone,
                             **kwargs).to(self.device)

    def get_cluster_result(self, dataloader, model, args):
        model.eval()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        b_all = torch.zeros((args.n_train, model.embedding_dim))
        for data in dataloader:
            with torch.no_grad():
                data = data.to(device)

                b = model.get_b(data.x, data.x_s, data.edge_index, data.batch, data.num_graphs)
                b_all[data.idx] = b.detach().cpu()
        cluster_result = run_kmeans(b_all.numpy(), args)
        return cluster_result

    def fit(self, dataset, args=None, label=None, dataloader=None,dataloader_val=None):
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model(**self.kwargs)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.model.train()
        self.decision_score_ = None
        self.train_dataloader = dataloader
        self.max_AUC = 0
        
        stop_counter = 0  # early stop counter
        N = 30  # early stop threshold

        for epoch in range(1, args.num_epoch + 1):
            if args.is_adaptive:
                if epoch == 1:
                    weight_b, weight_g, weight_n = 1, 1, 1
                else:
                    weight_b, weight_g, weight_n = std_b ** args.alpha, std_g ** args.alpha, std_n ** args.alpha
                    weight_sum = (weight_b + weight_g + weight_n) / 3
                    weight_b, weight_g, weight_n = weight_b / weight_sum, weight_g / weight_sum, weight_n / weight_sum

            if args.is_adaptive:
                loss_b_all, loss_g_all, loss_n_all = [], [], []
            y_score_all = []
            loss_all = 0
            for data in dataloader:
                data = data.to(device)
                optimizer.zero_grad()
                loss_epoch, score_epoch = self.forward_model(data, dataloader, args)
                loss_g, loss_b, loss_n = loss_epoch
                y_score_b, y_score_g, y_score_n = score_epoch
                if args.is_adaptive:
                    loss = weight_b * loss_b.mean() + weight_g * loss_g.mean() + weight_n * loss_n.mean()
                    loss_b_all = loss_b_all + loss_b.detach().cpu().tolist()
                    loss_g_all = loss_g_all + loss_g.detach().cpu().tolist()
                    loss_n_all = loss_n_all + loss_n.detach().cpu().tolist()
                else:
                    loss = loss_b.mean() + loss_g.mean() + loss_n.mean()
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()
                if args.is_adaptive:
                    mean_b, std_b = np.mean(loss_b_all), np.std(loss_b_all)
                    mean_g, std_g = np.mean(loss_g_all), np.std(loss_g_all)
                    mean_n, std_n = np.mean(loss_n_all), np.std(loss_n_all)
                # batch_size = data.batch_size
              
                if args.is_adaptive:
                    y_score = (y_score_b - mean_b) / std_b + (y_score_g - mean_g) / std_g + (y_score_n - mean_n) / std_n
                else:
                    y_score = y_score_b + y_score_g + y_score_n
                # print(y_score)
                y_score_all = y_score_all + y_score.detach().cpu().tolist()
            print('[TRAIN] Epoch:{:03d} | Loss:{:.4f}'.format(epoch, loss_all / args.n_train))
            if (epoch) % args.eval_freq == 0 and epoch > 0:
                self.model.eval()

                y_val = []
                score_val = []
                for data in dataloader_val:
                    
                    data = data.to(device)
                    emb, emb_list = self.model(data)
                    cluster_result = self.get_cluster_result(self.train_dataloader, self.model, args)
                    y_score_b, y_score_g, y_score_n = self.model.score_func(emb, emb_list, data, cluster_result)
                    y_score = y_score_b + y_score_g + y_score_n
                    
                    y_true = data.y
                    y_val = y_val + y_true.detach().cpu().tolist()
                    score_val = score_val + y_score.detach().cpu().tolist()


                val_auc = ood_auc(y_val,score_val)

                if val_auc > self.max_AUC:
                    self.max_AUC = val_auc
                    stop_counter = 0  # restart counter
                    torch.save(self.model, os.path.join(self.path, 'model_GOOD_D.pth'))
                else:
                    stop_counter += 1  
                
                if stop_counter >= N:
                    print(f'Early stopping triggered after {epoch} epochs due to no improvement in AUC for {N} consecutive evaluations.')
                    break  # early stop, jump out
                        # self.decision_score_[node_idx[:batch_size]] = y_score

        # self._process_decision_score()
        return self
    
    def is_directory_empty(self,directory):
        # list folder
        files_and_dirs = os.listdir(directory)
        # If the list is empty, the directory is empty
        return len(files_and_dirs) == 0
    
    def decision_function(self, dataset, label=None, dataloader=None, args=None):
        
        if self.is_directory_empty(self.path):
            print("Can't find the path")
        else:
            print("Loading Model Weight")
            self.model = torch.load(os.path.join(self.path,'model_GOOD_D.pth'))
        self.model.eval()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        y_score_all = []
        y_true_all = []
        for data in dataloader:
            data = data.to(device)
            emb, emb_list = self.model(data)
            cluster_result = self.get_cluster_result(self.train_dataloader, self.model, args)
            y_score_b, y_score_g, y_score_n = self.model.score_func(emb, emb_list, data, cluster_result)

            y_score = y_score_b + y_score_g + y_score_n
            # outlier_score[node_idx[:batch_size]] = y_score
            y_score_all = y_score_all + y_score.detach().cpu().tolist()
            y_true = data.y
            y_true_all = y_true_all + y_true.detach().cpu().tolist()
        return y_score_all, y_true_all

    def forward_model(self, dataset, dataloader=None, args=None):

        emb, emb_list = self.model(dataset)
        cluster_result = self.get_cluster_result(self.train_dataloader, self.model, args)
        loss, score = self.model.loss_func(emb, emb_list, dataset, cluster_result)

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

