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
       
    elif model_name == "GLADC":
        return GLADC(lr=args.lr,
                     num_epochs=args.num_epoch,
                     dropout=args.dropout,
                     batch_size=args.batch_size,
                     hidden_dim=args.hidden_dim,
                     output_dim = args.output_dim,
                     num_gc_layers = args.num_layer,
                     max_nodes_num=args.max_nodes_num,
                     feature_dim=args.dataset_num_features)
       
    elif model_name == "SIGNET":
        return SIGNET(num_epochs=args.num_epoch,
                     gpu=args.gpu,
                     lr=args.lr,
                     input_dim=args.dataset_num_features,
                     input_dim_edge=args.n_edge_feat,
                     args=args)

   
