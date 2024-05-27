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
                     batch_size=args.batch_size,
                     args=args)
    
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
                     grand=False,
                     args=args)

    elif model_name == "GLADC":
        return GLADC(lr=args.lr,
                     num_epochs=args.num_epoch,
                     dropout=args.dropout,
                     batch_size=args.batch_size,
                     hidden_dim=args.hidden_dim,
                     output_dim = args.output_dim,
                     num_gc_layers = args.num_layer,
                     max_nodes_num=args.max_nodes_num,
                     feature_dim=args.dataset_num_features,
                     DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     bn = args.bn,
                     args = args
                    )
    elif model_name == "SIGNET":
        return SIGNET(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     input_dim=args.dataset_num_features,
                     input_dim_edge=args.n_edge_feat,
                     args=args)
    elif model_name == "GLocalKD":
        return GLocalKD(args =args)
    elif model_name == "CVTGAD":
        return CVTGAD(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     args =args)
    
    elif  model_name == "OCGTL":
        return OCGTL(in_dim=args.dataset_num_features,
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
                     grand=False,
                     args=args)
    elif  model_name == "OCGTL":
        return OCGTL(in_dim=args.dataset_num_features,
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
                     grand=False,
                     args=args)
    elif model_name == "OCGIN":
        return OCGIN(in_dim=args.dataset_num_features,
                     hid_dim=args.hidden_dim,
                     num_layers=args.num_layer,
                     str_dim=args.dg_dim + args.rw_dim,
                     weight_decay=weight_decay,
                     dropout=args.dropout,
                     lr=args.lr,
                     n_train_data=args.n_train,
                     epoch=args.num_epoch,
                     gpu=args.gpu,
                     batch_size=args.batch_size,
                     grand=False,
                     args=args)
    elif model_name == "GraphCL_IF":
        return GraphCL_IF(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     detector = args.detector,
                     IF_n_trees =args.IF_n_trees,
                     IF_sample_ratio =args.IF_sample_ratio,
                     gamma =args.gamma,
                     nu=args.nuOCSVM,
                     args =args)
    elif model_name == "GraphCL_OCSVM":
        return GraphCL_OCSVM(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     detector = args.detector,
                     IF_n_trees =args.IF_n_trees,
                     IF_sample_ratio =args.IF_sample_ratio,
                     gamma =args.gamma,
                     nu=args.nuOCSVM,
                     args =args)
    elif model_name == "InfoGraph_IF":
        return InfoGraph_IF(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     detector = args.detector,
                     IF_n_trees =args.IF_n_trees,
                     IF_sample_ratio =args.IF_sample_ratio,
                     gamma =args.gamma,
                     nu=args.nuOCSVM,
                     args =args)
    elif model_name == "InfoGraph_OCSVM":
        return InfoGraph_OCSVM(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     detector = args.detector,
                     IF_n_trees =args.IF_n_trees,
                     IF_sample_ratio =args.IF_sample_ratio,
                     gamma =args.gamma,
                     nu=args.nuOCSVM,
                     args =args)
    elif model_name == "KernelGLAD":
        return KernelGLAD(DS = args.DS,
                     DS_pair=args.DS_pair,
                     exp_type = args.exp_type,
                     model_name = args.model,
                     detector = args.detector,
                     kernel= args.kernel,
                     WL_iter=args.WL_iter,
                     PK_bin_width=args.PK_bin_width,
                     LOF_n_neighbors=args.LOF_n_neighbors,
                     LOF_n_leaf=args.LOF_n_leaf,
                     IF_n_trees =args.IF_n_trees,
                     IF_sample_ratio =args.IF_sample_ratio,
                     detectorskernel='precomputed',
                     nu=args.nuOCSVM,
                     args =args)
   

