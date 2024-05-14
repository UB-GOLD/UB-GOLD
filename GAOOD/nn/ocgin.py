import torch.nn as nn
from ..nn.encoder import myGIN
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool
import torch

class OCGIN(nn.Module):
    def __init__(self, dim_features, args):
        super(OCGIN, self).__init__()

        self.dim_targets = args.hidden_dim
        self.num_layers = args.num_layer
        self.device = args.gpu
        self.net = myGIN(dim_features, self.dim_targets, args)
        self.center = torch.zeros(1, self.dim_targets * self.num_layers, requires_grad=False).to('cuda')

        self.ord = 2
        self.reset_parameters()
    def forward(self, data):
        data = data.to(self.device)
        z = self.net(data)
        return [z, self.center]

    def init_center(self, train_loader):
        with torch.no_grad():
            for data in train_loader:
                data = data.to('cuda')
                z = self.forward(data)
                self.center += torch.sum(z[0], 0, keepdim=True)
            self.center = self.center / len(train_loader.dataset)

    def reset_parameters(self):
        self.net.reset_parameters()

    def loss_func(self, z_c, eval=False):

        z = z_c[0] - z_c[1]
        diffs = torch.pow(z.norm(p=2, dim=-1), self.ord)
        if eval:

            return diffs
        else:
            return diffs