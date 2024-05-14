
import torch.nn as nn
from ..nn.encoder import myGIN
import torch.nn.init as init
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool,global_max_pool



import torch
import torch.nn.functional as F
class ocgtl(nn.Module):
    def __init__(self, dim_features,args):
        super(ocgtl, self).__init__()

        num_trans = args.num_trans
        dim_targets = args.hidden_dim
        num_layers = args.num_layer
        self.device = args.gpu
        self.gins = []
        for _ in range(num_trans):
            self.gins.append(myGIN(dim_features,dim_targets,args))
        self.gins = nn.ModuleList(self.gins)
        self.center = nn.Parameter(torch.empty(1, 1,dim_targets*num_layers), requires_grad=True)
        self.temp = 1.
        self.reset_parameters()
    def forward(self,data):
        data = data.to(self.device)
        z_cat = []
        for i,model in enumerate(self.gins):
            z = model(data)
            z_cat.append(z.unsqueeze(1))
        z_cat = torch.cat(z_cat,1)
        z_cat[:,0] = z_cat[:,0]+self.center[:,0]
        return [z_cat,self.center]

    def reset_parameters(self):
        init.normal_(self.center)
        for nn in self.gins:
            nn.reset_parameters()
    def loss_func(self, z_c,eval=False):

        z = z_c[0]
        c = z_c[1]

        z_norm = (z-c).norm(p=2, dim=-1)
        z = F.normalize(z, p=2, dim=-1)

        z_ori = z[:, 0]  # n,z
        z_trans = z[:, 1:]  # n,k-1, z
        batch_size, num_trans, z_dim = z.shape

        sim_matrix = torch.exp(torch.matmul(z, z.permute(0, 2, 1) / self.temp))  # n,k,k
        mask = (torch.ones_like(sim_matrix).to(z) - torch.eye(num_trans).unsqueeze(0).to(z)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(batch_size, num_trans, -1)
        trans_matrix = sim_matrix[:, 1:].sum(-1)  # n,k-1
        pos_sim =  torch.exp(torch.sum(z_trans * z_ori.unsqueeze(1), -1) / self.temp ) # n,k-1

        loss_tensor = torch.pow(z_norm[:,1:],1)+(torch.log(trans_matrix)-torch.log(pos_sim))

        if eval:
            score=loss_tensor.sum(1)
            return score
        else:
            loss=loss_tensor.sum(1)
            return loss

