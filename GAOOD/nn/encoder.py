

import torch
import torch.nn.functional as F



import torch
from torch.nn import Linear, ReLU, ModuleList, Sequential, BatchNorm1d
from torch_geometric.nn import GCNConv, GATConv, GINConv, SAGPooling, TopKPooling, BatchNorm, global_add_pool, global_mean_pool
import torch.nn.functional as F
from torch_geometric.utils import batched_negative_sampling, dropout_adj
from torch_scatter import scatter
import torch.nn as nn

import torch.nn.init as init

def create_model(backbone, in_channels, hid_channels, num_unit, dropout=0.0, dropedge=0.0, batch_norm=False):
    if backbone == 'GCN':
        model = GCN(in_channels, hid_channels, num_unit, dropout, dropedge, batch_norm=batch_norm)
    elif backbone == 'GAT':
        model = GAT(in_channels, hid_channels, num_unit, dropout, dropedge, batch_norm=batch_norm)
    elif backbone == 'GIN':
        model = GIN(in_channels, hid_channels, num_unit, dropout, dropedge, batch_norm=batch_norm)
    elif backbone == 'SAGPool':
        model = SAGPool(in_channels, hid_channels, num_unit, dropout, batch_norm=batch_norm)
    elif backbone == 'TopKPool':
        model = TopKPool(in_channels, hid_channels, num_unit, dropout, batch_norm=batch_norm)
    else:
        raise ValueError("Unknown backbone type!")
    return model

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, num_unit, dropout=0.0, dropedge=0.0, batch_norm=False):
        super(GCN, self).__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.dropedge = dropedge
        self.batch_norm = batch_norm

        in_conv = GCNConv(in_channels=in_channels, out_channels=hid_channels)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        self.convs.append(in_conv)
        for i in range(num_unit):
            conv = GCNConv(in_channels=hid_channels, out_channels=hid_channels)
            bn = BatchNorm(hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(bn)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, edge_index, *args):
        for i, conv in enumerate(self.convs[:-1]):
            edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
            x = conv(x, edge_index_)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
        x = self.convs[-1](x, edge_index_)
        return x, args[1]

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, num_unit, dropout=0.0, dropedge=0.0, batch_norm=False):
        super(GAT, self).__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.dropedge = dropedge
        self.batch_norm = batch_norm

        in_conv = GATConv(in_channels=in_channels, out_channels=hid_channels)
        self.convs = ModuleList()
        self.batch_norms = ModuleList()

        self.convs.append(in_conv)
        for i in range(num_unit):
            conv = GATConv(in_channels=hid_channels, out_channels=hid_channels)
            bn = BatchNorm(hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(bn)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, edge_index, *args):
        for i, conv in enumerate(self.convs[:-1]):
            edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
            x = conv(x, edge_index_)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
        x = self.convs[-1](x, edge_index_)
        return x, args[1]

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, num_unit, dropout=0.0, dropedge=0.0, batch_norm = False):
        super(GIN, self).__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.dropedge = dropedge
        self.batch_norm = batch_norm

        in_conv = GINConv(Sequential(Linear(in_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
        self.convs =  ModuleList()
        self.batch_norms = ModuleList()

        self.convs.append(in_conv)
        for i in range(num_unit):
            conv = GINConv(Sequential(Linear(hid_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
            bn = BatchNorm(hid_channels)
            self.convs.append(conv)
            self.batch_norms.append(bn)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, edge_index, *args):
        for i, conv in enumerate(self.convs[:-1]):
            edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
            x = conv(x, edge_index_)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        edge_index_, _ = dropout_adj(edge_index, p=self.dropedge)
        x = self.convs[-1](x, edge_index_)
        return x, args[1]

class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, num_unit, dropout=0.0, min_score=0.001, batch_norm=False):
        super(SAGPool, self).__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.batch_norm = batch_norm

        in_conv = GINConv(Sequential(Linear(in_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
        self.convs = ModuleList()
        self.pools = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(in_conv)

        for i in range(num_unit):
            conv = GINConv(Sequential(Linear(hid_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
            pool = SAGPooling(hid_channels, min_score=min_score, GNN=GCNConv)
            bn = BatchNorm(hid_channels)
            self.convs.append(conv)
            self.pools.append(pool)
            self.batch_norms.append(bn)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_index, edge_attr, batch, _, _ = self.pools[i](x, edge_index, edge_attr, batch)
        x = self.convs[-1](x, edge_index)
        return x, batch

class TopKPool(torch.nn.Module):
    def __init__(self, in_channels, hid_channels, num_unit, dropout=0.0, min_score=0.001, batch_norm=False):
        super(TopKPool, self).__init__()

        self.num_unit = num_unit
        self.dropout = dropout
        self.batch_norm = batch_norm

        in_conv = GINConv(Sequential(Linear(in_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
        self.convs = ModuleList()
        self.pools = ModuleList()
        self.batch_norms = ModuleList()
        self.convs.append(in_conv)

        for i in range(num_unit):
            conv = GINConv(Sequential(Linear(hid_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels)))
            pool = TopKPooling(hid_channels, min_score=min_score)
            bn = BatchNorm(hid_channels)
            self.convs.append(conv)
            self.pools.append(pool)
            self.batch_norms.append(bn)

        self.activation = F.relu

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for bn in self.batch_norms:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
            if self.batch_norm:
                x = self.batch_norms[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x, edge_index, edge_attr, batch, _, _ = self.pools[i](x, edge_index, edge_attr, batch)
        x = self.convs[-1](x, edge_index)
        return x, batch

class MLP(torch.nn.Module):
    def __init__(self,
                num_features,
                num_classes,
                hidden_size,
                dropout=0.5,
                activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = Linear(num_features, hidden_size)
        self.fc2 = Linear(hidden_size, num_classes)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x = data.x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class CosineLSM(torch.nn.Module):
    """One of instantiations for our structure estimation model"""
    def __init__(self, in_channels, hid_channels, dropout=0.0, neg_ratio=1.0, m=1):
        super(CosineLSM, self).__init__()
        self.m = m
        self.x_encs = Linear(in_channels, m * hid_channels)
        self.activation = F.relu
        self.cosine = F.cosine_similarity
        self.neg_ratio = neg_ratio
        self.dropout = dropout
        self.hid_channels = hid_channels

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.x_encs(x))

        # positive edges
        x_query = F.embedding(edge_index[0], x).view(-1, self.m, self.hid_channels)
        x_key = F.embedding(edge_index[1], x).view(-1, self.m, self.hid_channels)
        e_pred_similarity = self.cosine(x_query, x_key, dim=2)
        e_pred_similarity = (e_pred_similarity + torch.ones_like(e_pred_similarity)) / 2 \
                            + torch.ones_like(e_pred_similarity) * 0.0001 # normalize to [0, 1]
        e_pred_pos = torch.mean(e_pred_similarity, dim=1)

        # negative edges
        e_pred_neg = None
        edge_index_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = edge_index.size(1)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = batched_negative_sampling(edge_index, batch, num_edges_neg)
            edge_index_neg = edge_index_neg.long()
            x_query = F.embedding(edge_index_neg[0], x).view(-1, self.m, self.hid_channels)
            x_key = F.embedding(edge_index_neg[1], x).view(-1, self.m, self.hid_channels)
            e_pred_similarity = self.cosine(x_query, x_key, dim=2)
            e_pred_similarity = (e_pred_similarity + torch.ones_like(e_pred_similarity)) / 2 - torch.ones_like(e_pred_similarity) * 0.0001
            e_pred_neg = torch.mean(e_pred_similarity, dim=1)

        return e_pred_pos, e_pred_neg, edge_index_neg

    def get_reg_loss(self, x, edge_index, batch):
        e_pred_pos, e_pred_neg, edge_index_neg = self.forward(x, edge_index, batch)
        e_logprob_pos = -(e_pred_pos.log()) # calculate negative log likelihood
        batch_num = torch.max(batch).item() + 1
        nll_p_g_x = scatter(e_logprob_pos, batch[edge_index[0]], dim_size=batch_num, reduce='mean') # scatter nll to each batch
        if e_pred_neg is not None:
            e_logprob_neg = -((1-e_pred_neg).log())
            nll_p_g_x += scatter(e_logprob_neg, batch[edge_index_neg[0]], dim_size=batch_num, reduce='mean')
            nll_p_g_x = nll_p_g_x / 2

        return nll_p_g_x

class LSM(torch.nn.Module):
    """One of instantiations for our structure estimation model"""
    def __init__(self, in_channels, hid_channels, dropout=0.0, neg_ratio=1.0):
        super(LSM, self).__init__()
        self.x_enc = Sequential(Linear(in_channels, hid_channels), ReLU(), Linear(hid_channels, hid_channels))
        self.p_e_x = Linear(2 * hid_channels, 1)
        self.dropout = dropout
        self.activation = F.relu
        self.neg_ratio = neg_ratio

    def forward(self, x, edge_index, batch):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.x_enc(x))

        # positive edges
        x_query = F.embedding(edge_index[0], x)
        x_key = F.embedding(edge_index[1], x)
        e_pred_pos = self.p_e_x(torch.cat([x_query, x_key], dim=1))

        # sample negative edges
        e_pred_neg = None
        edge_index_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = edge_index.size(1)
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = batched_negative_sampling(edge_index, batch, num_edges_neg)

            x_query = F.embedding(edge_index_neg[0], x)
            x_key = F.embedding(edge_index_neg[1], x)
            e_pred_neg = self.p_e_x(torch.cat([x_query, x_key], dim=1))

        return e_pred_pos, e_pred_neg, edge_index_neg

    def get_reg_loss(self, x, edge_index, batch):
        e_pred_pos, e_pred_neg, edge_index_neg = self.forward(x, edge_index, batch)
        e_logprob_pos = torch.squeeze(-F.logsigmoid(e_pred_pos)) # calculate negative log likelihood
        batch_num = torch.max(batch).item() + 1
        nll_p_g_x = scatter(e_logprob_pos, batch[edge_index[0]], dim_size=batch_num, reduce='mean') # scatter nll to each batch
        if e_pred_neg is not None:
            e_logprob_neg = torch.squeeze(-F.logsigmoid(-e_pred_neg))
            nll_p_g_x += scatter(e_logprob_neg, batch[edge_index_neg[0]], dim_size=batch_num, reduce='mean')
            nll_p_g_x = nll_p_g_x / 2

        return nll_p_g_x
class GNA(torch.nn.Module):
    """
    Graph Node Attention Network (GNA). See :cite:`yuan2021higher` for
    more details.
    """
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_layers,
                 out_channels,
                 dropout,
                 act):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GNAConv(in_channels, hidden_channels))
        for layer in range(num_layers - 2):
            self.layers.append(GNAConv(hidden_channels,
                                       hidden_channels))
        self.layers.append(GNAConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.act = act

    def forward(self, s, edge_index):
        """
        Forward computation.

        Parameters
        ----------
        s : torch.Tensor
            Input node embeddings.
        edge_index : torch.Tensor
            Edge index.

        Returns
        -------
        s : torch.Tensor
            Updated node embeddings.
        """
        for layer in self.layers:
            s = layer(s, edge_index)
            s = F.dropout(s, self.dropout, training=self.training)
            if self.act is not None:
               s = self.act(s)
        return s

class GraphNorm(nn.Module):
    def __init__(self, dim, affine=True):
        super(GraphNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(dim),requires_grad=affine)
        self.bias = nn.Parameter(torch.zeros(dim),requires_grad=False)
        self.scale = nn.Parameter(torch.ones(dim),requires_grad=affine)
    def forward(self,node_emb,graph):
        try:
            num_nodes_list = torch.tensor(graph.__num_nodes_list__).long().to(node_emb.device)
        except:
            num_nodes_list = graph.ptr[1:]-graph.ptr[:-1]

        graph_batch_size = graph.batch.max().item() + 1
        num_nodes_list = num_nodes_list.long().to(node_emb.device)
        node_mean = scatter(node_emb, graph.batch, dim=0, dim_size=graph_batch_size, reduce='mean')
        node_mean = node_mean.repeat_interleave(num_nodes_list, 0)

        sub = node_emb - node_mean*self.scale
        node_std = scatter(sub.pow(2), graph.batch, dim=0, dim_size=graph_batch_size, reduce='mean')
        node_std = torch.sqrt(node_std + 1e-8)
        node_std = node_std.repeat_interleave(num_nodes_list, 0)
        norm_node = self.weight * sub / node_std + self.bias
        return norm_node
    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)
        init.ones_(self.scale)

class myMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, bias=True):
        super(myMLP, self).__init__()
        self.lin1 = Linear(in_dim, hidden, bias=bias)
        self.lin2 = Linear(hidden, out_dim, bias=bias)

    def forward(self, z):
        z = self.lin2(F.relu(self.lin1(z)))
        return z
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

class myGIN(torch.nn.Module):

    def __init__(self, dim_features, dim_targets, args):
        super(myGIN, self).__init__()

        hidden_dim = args.hidden_dim
        self.num_layers = args.num_layer
        self.nns = []
        self.convs = []
        self.norms = []
        self.projs = []
        self.use_norm = args.norm_layer
        bias = args.bias

        if args.aggregation == 'add':
            self.pooling = global_add_pool
        elif args.aggregation == 'mean':
            self.pooling = global_mean_pool

        for layer in range(self.num_layers):
            if layer == 0:
                input_emb_dim = dim_features
            else:
                input_emb_dim = hidden_dim
            self.nns.append(Sequential(Linear(input_emb_dim, hidden_dim, bias=bias), ReLU(),
                                       Linear(hidden_dim, hidden_dim, bias=bias)))
            self.convs.append(GINConv(self.nns[-1], train_eps=bias))  # Eq. 4.2
            if self.use_norm == 'gn':
                self.norms.append(GraphNorm(hidden_dim, True))
            self.projs.append(myMLP(hidden_dim, hidden_dim, dim_targets, bias))

        self.nns = nn.ModuleList(self.nns)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.projs = nn.ModuleList(self.projs)

    def forward(self, graph):
        x, edge_index, batch = graph.x, graph.edge_index, graph.batch
        z_cat = []

        for layer in range(self.num_layers):
            x = self.convs[layer](x, edge_index)
            if self.use_norm == 'gn':
                x = self.norms[layer](x, graph)
            x = F.relu(x)
            z = self.projs[layer](x)
            z = self.pooling(z, batch)
            z_cat.append(z)
        z_cat = torch.cat(z_cat, -1)
        return z_cat

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for proj in self.projs:
            proj.reset_parameters()

class myGIN_classifier(nn.Module):
    def __init__(self, dim_features, dim_targets, args):
        super(myGIN_classifier, self).__init__()

        hidden_dim = args.hidden_dim
        self.num_layers = args.num_layer
        self.nns = []
        self.convs = []
        self.norms = []
        self.projs = []
        self.use_norm = args.norm_layer
        bias = args.bias

        if args.aggregation == 'add':
            self.pooling = global_add_pool
        elif args.aggregation == 'mean':
            self.pooling = global_mean_pool

        for layer in range(self.num_layers):
            if layer == 0:
                input_emb_dim = dim_features
            else:
                input_emb_dim = hidden_dim
            self.nns.append(Sequential(Linear(input_emb_dim, hidden_dim, bias=bias), ReLU(),
                                       Linear(hidden_dim, hidden_dim, bias=bias)))
            self.convs.append(GINConv(self.nns[-1], train_eps=bias))  # Eq. 4.2
            if self.use_norm == 'gn':
                self.norms.append(GraphNorm(hidden_dim, True))
            self.projs.append(Linear(hidden_dim, dim_targets, bias=bias))

        self.nns = nn.ModuleList(self.nns)
        self.convs = nn.ModuleList(self.convs)
        self.norms = nn.ModuleList(self.norms)
        self.projs = nn.ModuleList(self.projs)


    def forward(self,graph):
        x,edge_index,batch = graph.x,graph.edge_index,graph.batch

        y_pred = 0
        for layer in range(self.num_layers):
            x = self.convs[layer](x, edge_index)
            if self.use_norm == 'gn':
                x = self.norms[layer](x, graph)
            x = F.relu(x)
            y = F.dropout(self.projs[layer](self.pooling(x,batch)),p=self.dropout)
            y_pred +=y

        return y_pred

    def reset_parameters(self):
        for norm in self.norms:
            norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for proj in self.projs:
            proj.reset_parameters()
