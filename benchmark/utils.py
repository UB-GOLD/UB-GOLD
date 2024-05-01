from random import choice
from GAOOD.detector import *

from torch_geometric.nn import MLP
from sklearn.ensemble import IsolationForest


def init_model(args):
   
    model_name = args.model
    gpu = args.gpu

    
    if model_name == "GOOD-D":
        return GOOD_D(args.dataset_num_features,args.hidden_dim, args.num_layer, args.dg_dim+args.rw_dim,
                      weight_decay=weight_decay,
                     dropout=0.5,
                     lr=args.lr,
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=args.batch_size,
                     num_neigh=num_neigh)
    
    elif model_name == "GraphDE":
        
        return GraphDE(in_dim=args.dataset_num_features,hid_dim=args.hidden_dim,
                       num_layers=args.num_layer,
                       str_dim=args.dg_dim+args.rw_dim,
                       weight_decay=weight_decay,
                     dropout=0.5,
                     lr=args.lr,
                     n_train_data = args.n_train,
                     epoch=epoch,
                     gpu=gpu,
                     batch_size=args.batch_size,
                     num_neigh=num_neigh,grand=False)

    elif model_name == "GLocalKD":

        return GLocalKD(in_dim=args.dataset_num_features,
                        max_nodes_num=args.max_nodes_num)