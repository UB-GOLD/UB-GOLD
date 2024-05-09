from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, GCNConv, GATConv, global_mean_pool, global_add_pool, global_max_pool
import torch
import torch.nn.functional as F
import torch.nn as nn


# -------------------------------------------------- definition CVTGAD----------------------------------------------------------#
class CVTGAD(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, feat_dim, str_dim, args):
        super(CVTGAD, self).__init__()

        self.embedding_dim = hidden_dim * num_gc_layers

        if args.GNN_Encoder == 'GCN':
            self.encoder_feat = Encoder_GCN(feat_dim, hidden_dim, num_gc_layers, args)
            self.encoder_str = Encoder_GCN(str_dim, hidden_dim, num_gc_layers, args)
        elif args.GNN_Encoder == 'GIN':
            self.encoder_feat = Encoder_GIN(feat_dim, hidden_dim, num_gc_layers, args)
            self.encoder_str = Encoder_GIN(str_dim, hidden_dim, num_gc_layers, args)
        elif args.GNN_Encoder == 'GAT':
            self.encoder_feat = Encoder_GAT(feat_dim, hidden_dim, num_gc_layers, args)
            self.encoder_str = Encoder_GAT(str_dim, hidden_dim, num_gc_layers, args)

        self.proj_head_feat_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                              nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_g = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                             nn.Linear(self.embedding_dim, self.embedding_dim))

        self.proj_feat_g_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.proj_str_g_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.Cross_Attention_g = Cross_Attention(self.embedding_dim, self.embedding_dim, args=args)

        self.proj_head_feat_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                              nn.Linear(self.embedding_dim, self.embedding_dim))
        self.proj_head_str_n = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.ReLU(inplace=True),
                                             nn.Linear(self.embedding_dim, self.embedding_dim))

        self.proj_feat_n_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.proj_str_n_Transformer = Transformer(self.embedding_dim, self.embedding_dim)
        self.Cross_Attention_n = Cross_Attention(self.embedding_dim, self.embedding_dim, args=args)

        self.proj_head_b = nn.Sequential(nn.Linear(self.embedding_dim * 2, self.embedding_dim), nn.ReLU(inplace=True),
                                         nn.Linear(self.embedding_dim, self.embedding_dim))

        self.proj_b_CrossAttention = Transformer(self.embedding_dim * 2, self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def get_b(self, x_f, x_s, edge_index, batch, num_graphs):
        g_f, _ = self.encoder_feat(x_f, edge_index, batch)
        g_s, _ = self.encoder_str(x_s, edge_index, batch)
        b = self.proj_head_b(torch.cat((g_f, g_s), 1))
        return b

    def forward(self, x_f, x_s, edge_index, batch, num_graphs):

        g_f, n_f = self.encoder_feat(x_f, edge_index, batch)
        g_s, n_s = self.encoder_str(x_s, edge_index, batch)
        # return g_f, g_s, n_f, n_s

        # b = self.proj_head_b(torch.cat((g_f, g_s), 1))
        # b = self.proj_b_CrossAttention(torch.cat((g_f, g_s), 1))

        g_f_1 = self.proj_head_feat_g(g_f)
        g_s_1 = self.proj_head_str_g(g_s)
        # g_f_1 = g_f
        # g_s_1 = g_s

        # g_f = self.proj_feat_g_Transformer(g_f)
        # g_s = self.proj_str_g_Transformer(g_s)
        g_f_2, g_s_2 = self.Cross_Attention_g(g_f, g_s)
        # g_f_2, g_s_2 = g_f, g_s

        g_ff = g_f_1 + g_f_2
        g_ss = g_s_1 + g_s_2

        n_f_1 = self.proj_head_feat_n(n_f)
        n_s_1 = self.proj_head_str_n(n_s)
        # n_f_1 = n_f
        # n_s_1 = n_s

        # n_f = self.proj_feat_n_Transformer(n_f)
        # n_s = self.proj_str_n_Transformer(n_s)
        n_f_2, n_s_2 = self.Cross_Attention_n(n_f, n_s)
        # n_f_2, n_s_2 = n_f, n_s

        n_ff = n_f_1 + n_f_2
        n_ss = n_s_1 + n_s_2

        # return b, g_f, g_s, n_f, n_s
        return g_f_2, g_s_2, n_f_2, n_s_2  #
        # return g_f_1, g_s_1, n_f_1, n_s_1
        # return g_ff, g_ss, n_ff, n_ss

    @staticmethod
    def scoring_b(b, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result[
            'density']

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)
        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))

        v, id = torch.min(sim_matrix, 1)

        return v

    @staticmethod
    def calc_loss_b(b, index, cluster_result, temperature=0.2):
        im2cluster, prototypes, density = cluster_result['im2cluster'], cluster_result['centroids'], cluster_result[
            'density']
        pos_proto_id = im2cluster[index].cpu().tolist()

        batch_size, _ = b.size()
        b_abs = b.norm(dim=1)
        prototypes_abs = prototypes.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', b, prototypes) / torch.einsum('i,j->ij', b_abs, prototypes_abs)
        sim_matrix = torch.exp(sim_matrix / (temperature * density))
        pos_sim = sim_matrix[range(batch_size), pos_proto_id]

        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss + 1e-12)
        return loss

    @staticmethod
    def calc_loss_n(x, x_aug, batch, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        node_belonging_mask = batch.repeat(batch_size, 1)
        node_belonging_mask = node_belonging_mask == node_belonging_mask.t()

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature) * node_belonging_mask
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        loss = global_mean_pool(loss, batch)

        return loss

    @staticmethod
    def calc_loss_g(x, x_aug, temperature=0.2):
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / temperature)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim)
        loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)

        loss_0 = - torch.log(loss_0)
        loss_1 = - torch.log(loss_1)
        loss = (loss_0 + loss_1) / 2.0
        return loss


class Encoder_GIN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GIN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            self.convs.append(conv)

        self.Transformer = Transformer(num_features, num_features)
        self.pool_type = args.graph_level_pool

    def forward(self, x, edge_index, batch):

        xs = []
        # x = self.Transformer(x)
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


class Encoder_GCN(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GCN, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GCNConv(dim, dim)
                # conv = GATConv(dim, dim)
            else:
                conv = GCNConv(num_features, dim)
                # conv = GATConv(num_features, dim)
            self.convs.append(conv)

        self.Transformer = Transformer(num_features, num_features)
        self.pool_type = args.graph_level_pool

    def forward(self, x, edge_index, batch):
        xs = []
        # x = self.Transformer(x)
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


class Encoder_GAT(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, args):
        super(Encoder_GAT, self).__init__()

        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()

        for i in range(num_gc_layers):
            if i:
                conv = GATConv(dim, dim)
            else:
                conv = GATConv(num_features, dim)
            self.convs.append(conv)

        self.Transformer = Transformer(num_features, num_features)
        self.pool_type = args.graph_level_pool

    def forward(self, x, edge_index, batch):
        xs = []
        # x = self.Transformer(x)
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, edge_index))
            xs.append(x)

        if self.pool_type == 'global_max_pool':
            xpool = [global_max_pool(x, batch) for x in xs]
        else:
            xpool = [global_mean_pool(x, batch) for x in xs]

        x = torch.cat(xpool, 1)
        return x, torch.cat(xs, 1)


########### Transformer
class Transformer(nn.Module):
    def __init__(self, attributed_dim, n_h) -> None:
        super().__init__()

        # c
        self.feats_channels = attributed_dim
        self.attention_channels = attributed_dim * 2
        # self.attention_channels = 1024
        self.fc_cat = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels)
        )

        self.w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        self.add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )

        # self.n_h = n_h

    def forward(self, features):
        features = features.unsqueeze(0)

        cat_feat = self.fc_cat(features)
        residual_feat = self.fc1(cat_feat)
        # residual_feat = cat_feat

        Q = self.w_qs(cat_feat)
        K = self.w_ks(cat_feat).permute(0, 2, 1)
        V = self.w_vs(cat_feat)
        attn = Q @ K
        attn = torch.softmax(attn, 2)
        atteneion = attn / (1e-9 + attn.sum(dim=1, keepdims=True))
        sc = atteneion @ V
        s = self.add_norm1(residual_feat + sc)
        ffn = self.add_norm2(s + self.fc_ffn(s))

        ffn = s + self.fc_ffn(s)
        output = self.fc3(ffn)

        output = output.squeeze(0)
        return output


################### Simplified Transformer with Cross-View Attention
class Cross_Attention(nn.Module):
    def __init__(self, attributed_dim, n_h,args) -> None:
        super().__init__()
        self.device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
        self.feats_channels = attributed_dim
        self.attention_channels = attributed_dim
        # self.attention_channels = 1024
        self.k = torch.sqrt(torch.FloatTensor([n_h])).to(self.device) #.cuda()

        # ------------------------------------------- view A --------------------------------------#
        self.A_projection_network = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.A_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
        )

        self.A_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.A_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.A_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.A_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        self.A_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.A_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.A_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.A_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )

        # ------------------------------------------- view B --------------------------------------#
        self.B_projection_network = nn.Sequential(
            nn.Linear(self.feats_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels)
        )
        self.B_residual_block = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, self.attention_channels),
            # nn.ReLU(),
            # nn.Linear(self.attention_channels, self.attention_channels),
        )
        ################# attention
        self.B_w_qs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.B_w_ks = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.B_w_vs = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.B_layer_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_layer_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)

        ################ FFN
        self.B_add_norm1 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_add_norm2 = nn.LayerNorm(self.attention_channels, eps=1e-6)
        self.B_fc_ffn = nn.Linear(self.attention_channels, self.attention_channels, bias=False)
        self.B_fc2 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels // 2),
            nn.ReLU(),
            nn.Linear(self.attention_channels // 2, self.attention_channels)
        )
        self.B_fc3 = nn.Sequential(
            nn.Linear(self.attention_channels, self.attention_channels),
            nn.ReLU(),
            nn.Linear(self.attention_channels, n_h)
        )

    def forward(self, feat_a, feat_b):
        # ----------------------------------------#
        feat_a = feat_a.unsqueeze(0)
        feat_b = feat_b.unsqueeze(0)

        # ----------------------------------------#
        A_feat = self.A_projection_network(feat_a)
        A_residual_feat = self.A_residual_block(A_feat)

        B_feat = self.B_projection_network(feat_b)
        B_residual_feat = self.B_residual_block(B_feat)

        # #----------------------------------------#
        A_Q = self.A_w_qs(A_feat)
        A_K = self.A_w_ks(A_feat).permute(0, 2, 1)
        A_V = self.A_w_vs(A_feat)

        B_Q = self.B_w_qs(B_feat)
        B_K = self.B_w_ks(B_feat).permute(0, 2, 1)
        B_V = self.B_w_vs(B_feat)

        # #--------------- Cross-View Attention ------------------#
        A_attn = A_Q @ B_K
        B_attn = B_Q @ A_K
        # A_Q, B_Q = B_Q, A_Q

        A_attn /= self.k
        B_attn /= self.k

        # ----------------------------------------#
        A_attn = torch.softmax(A_attn, 2)
        A_attention = A_attn / (1e-9 + A_attn.sum(dim=1, keepdims=True))
        A_sc = A_attention @ A_V
        # A_sc = A_atteneion @ B_V
        # A_sc = A_sc.permute(0, 2, 1)

        B_attn = torch.softmax(B_attn, 2)
        B_attention = B_attn / (1e-9 + B_attn.sum(dim=1, keepdims=True))
        B_sc = B_attention @ B_V

        # ----------------------------------------#
        A_s = self.A_add_norm1(A_residual_feat + A_sc)
        # A_ffn = self.A_add_norm2(A_s + self.A_fc_ffn(A_s))
        A_ffn = A_s + self.A_fc_ffn(A_s)
        # A_ffn = A_s

        B_s = self.B_add_norm1(B_residual_feat + B_sc)
        # B_ffn = self.B_add_norm2(B_s + self.B_fc_ffn(B_s))
        B_ffn = B_s + self.B_fc_ffn(B_s)
        # B_ffn = B_s

        # ---------------------------------------#

        # A_output = self.A_fc3(A_ffn)
        A_output = A_ffn

        # B_output = self.B_fc3(B_ffn)
        B_output = B_ffn

        # ---------------------------------------#
        A_output = A_output.squeeze(0)
        B_output = B_output.squeeze(0)
        return A_output, B_output


