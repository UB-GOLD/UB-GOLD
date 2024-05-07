from random import choice
from GAOOD.detector import *

from torch_geometric.nn import MLP
from sklearn.ensemble import IsolationForest


def init_model(args):
    weight_decay = 0.01
    model_name = args.model
   
    
    if model_name == "GOOD-D":
        return GOOD_D(in_dim = args.dataset_num_features, hid_dim = args.hidden_dim, 
                      num_layers = args.num_layer,
                      str_dim = args.dg_dim+args.rw_dim,
                      weight_decay=weight_decay,
                     dropout=args.dropout,
                     lr=args.lr,
                     epoch=args.num_epoch,
                     gpu=args.gpu,
                     batch_size=args.batch_size)
    
    elif model_name == "GraphDE":
        
        return GraphDE(in_dim=args.dataset_num_features,
                       hid_dim=args.hidden_dim,
                       num_layers=args.num_layer,
                       str_dim=args.dg_dim+args.rw_dim,
                       weight_decay=weight_decay,
                     dropout=args.dropout,
                     lr=args.lr,
                     n_train_data = args.n_train,
                     epoch=args.num_epoch,
                     gpu=args.gpu,
                     batch_size=args.batch_size,
                     grand=False)

    elif model_name == "GLocalKD":

        return GLocalKD(in_dim=args.dataset_num_features,
                        max_nodes_num=args.max_nodes_num,
                       num_epochs=args.num_epoch)
       
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

   
