
import torch
import torch.nn as nn
from torch_geometric.nn import GCN


class DotProductDecoder(nn.Module):
    

    def __init__(self,
                 in_dim,
                 hid_dim=64,
                 num_layers=1,
                 dropout=0.,
                 act=torch.nn.functional.relu,
                 sigmoid_s=False,
                 backbone=GCN,
                 **kwargs):
        super(DotProductDecoder, self).__init__()

        self.sigmoid_s = sigmoid_s
        self.nn = backbone(in_channels=in_dim,
                           hidden_channels=hid_dim,
                           num_layers=num_layers,
                           out_channels=hid_dim,
                           dropout=dropout,
                           act=act,
                           **kwargs)

    def forward(self, x, edge_index):
        
        h = self.nn(x, edge_index)
        s_ = h @ h.T
        if self.sigmoid_s:
            s_ = torch.sigmoid(s_)
        return s_
