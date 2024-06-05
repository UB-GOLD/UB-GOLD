import tqdm
import torch
import argparse
import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import warnings
from GAOOD.metric import *

from utils import init_model
from dataloader.data_loader import *
import argparse
import time
import pandas
from copy import deepcopy
from search_utils import *
import warnings

warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

import os
import numpy as np
import torch
import random

models = model_detector_dict.keys()


def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def main(args):
    columns = ['name']
    new_row = {}

    models = model_detector_dict.keys()
    if args.exp_type == 'ad':
         datasets=['AIDS','BZR','COLLAD','COX2','DD','DHFR','ENZYMES','Tox21_HSE','IMDB-BINARY','Tox21_MMP','NCI1','Tox21_p53','REDDIT-BINARY', 'Tox21_PPAR-gamma', 'PROTEINS_full' ]
    if args.exp_type == 'oodd':
        datasets = ['AIDS+DHFR', 'ogbg-molbbbp+ogbg-molbace', 'BZR+COX2', 'ogbg-molclintox+ogbg-mollipo',
                    'ENZYMES+PROTEINS', 'ogbg-molfreesolv+ogbg-moltoxcast', 'IMDB-MULTI+IMDB-BINARY', 'PTC_MR+MUTAG',
                    'ogbg-molesol+ogbg-molmuv', 'ogbg-moltox21+ogbg-molsider']
        #datasets = ['AIDS+DHFR', 'BZR+COX2']
    elif args.exp_type == 'ood':
        datasets=['GOODHIV+scaffold+concept',]#'GOODHIV+size+concept','GOODPCBA+scaffold+concept','GOODPCBA+size+concept','DrugOOD+ic50_size','DrugOOD+ic50_scaffold','DrugOOD+ec50_size','DrugOOD+ec50_scaffold']
    for dataset_name in datasets:
        for metric in ['AUROC', 'AUPRC','FPR95']:
            columns.append(dataset_name + '-' + metric)

    results = pandas.DataFrame(columns=columns)
    best_model_configs = {}
    file_id = None
    auc, ap, rec = [], [], []

    for model in models:
        model_result = {'name': model}
        best_model_configs[model] = {}

        for dataset_name in datasets:
            if args.exp_type == 'oodd':
                args.DS_pair = dataset_name
            elif args.exp_type == 'ood':
                args.DS = dataset_name
            elif args.exp_type == 'ad':
                args.DS = dataset_name
            best_val_score = 0
            for t in tqdm.tqdm(range(args.num_trial)):

                if args.exp_type == 'ad':
                    if args.DS.startswith('Tox21'):
                        dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_Tox21(
                            args)
                    else:
                        splits = get_ad_split_TU(args, fold=args.num_trial)
                if args.exp_type == 'oodd':
                    dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset(
                        args)
                elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
                    dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_TU(
                        args, splits[t])
                elif args.exp_type == 'ood':
                    print("-------")
                    print(args.exp_type)
                    dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset_spilt(
                        args)
                if args.model == 'GOOD-D':
                    
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'GraphDE':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'CVTGAD':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    args.GNN_Encoder = model_config['GNN_Encoder']
                    args.graph_level_pool = model_config['graph_level_pool']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'GLADC':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    args.nobias = model_config['nobias']
                    args.nobn = model_config['nobn']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'SIGNET':
                    model_config = sample_param(args.model, args.DS, t+1)
                    args.hidden_dim = model_config['h_feats']
                    args.encoder_layers = model_config['encoder_layers']
                    args.lr = model_config['lr']
                    args.n_edge_feat = meta['num_edge_feat']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    args.pooling = model_config['pooling']
                    args.graph_level_pool = model_config['graph_level_pool']
                    args.readout = model_config['readout']
                    args.explainer_model =  model_config['explainer_model']
                    args.explainer_layers = model_config['explainer_layers']
                    args.explainer_hidden_dim = model_config['explainer_hidden_dim']
                    args.explainer_readout = model_config['explainer_readout']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'GLocalKD':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    args.clip = model_config['clip']
                    args.nobn = model_config['nobn']
                    args.nobias = model_config['nobias']
                    args.output_dim = model_config['output_dim']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'OCGTL':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'OCGIN':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.hidden_dim = model_config['h_feats']
                    args.num_layer = model_config['num_layers']
                    args.lr = model_config['lr']
                    args.dropout = model_config['drop_rate']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'GraphCL_IF':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.IF_n_trees = model_config['n_trees']
                    args.IF_sample_ratio = model_config['sample_ratio']
                    args.lr = model_config['lr']
                    
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'GraphCL_OCSVM':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.IF_n_trees = model_config['n_trees']
                    args.IF_sample_ratio = model_config['sample_ratio']
                    args.lr = model_config['lr']
                    
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'InforGraph_IF':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.IF_n_trees = model_config['n_trees']
                    args.IF_sample_ratio = model_config['sample_ratio']
                    args.lr = model_config['lr']
                    
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'InforGraph_OCSVM':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.IF_n_trees = model_config['n_trees']
                    args.IF_sample_ratio = model_config['sample_ratio']
                    args.lr = model_config['lr']
                    
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
                elif args.model == 'KernelGLAD':
                    model_config = sample_param(args.model, args.DS, t+1)
                    print(model_config)
                    args.IF_n_trees = model_config['n_trees']
                    args.IF_sample_ratio = model_config['sample_ratio']
                    args.lr = model_config['lr']
                    args.LOF_n_neighbors = model_config['n_neighbors']
                    args.LOF_n_leaf = model_config['n_leaf']
                    args.WL_iter = model_config['WL_iter']
                    args.dataset_num_features = meta['num_feat']
                    args.n_train =  meta['num_train']
            
                    model = init_model(args)
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        print(model_config)
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, best95 = ood_auc(pred,score), ood_aupr(pred,score),fpr95(pred, score)
            model_result[dataset_name + '-AUROC'] = best_troc
            model_result[dataset_name + '-AUPRC'] = best_tprc
            model_result[dataset_name + '-FPR95'] = best95
        model_result = pandas.DataFrame(model_result, index=[0])
        results = pandas.concat([results, model_result])
        file_id = save_results(results, file_id)





    # ap = torch.tensor(ap)
    # rec = torch.tensor(rec)

    '''
    print(args.dataset + " " + model.__class__.__name__ + " " +
          "AUC: {:.4f}±{:.4f} ({:.4f})\t"
          "AP: {:.4f}±{:.4f} ({:.4f})\t"
          "Recall: {:.4f}±{:.4f} ({:.4f})".format(torch.mean(auc),
                                                  torch.std(auc),
                                                  torch.max(auc),
                                                  torch.mean(ap),
                                                  torch.std(ap),
                                                  torch.max(ap),
                                                  torch.mean(rec),
                                                  torch.std(rec),
                                                  torch.max(rec)))
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="GOOD-D",
                        help="supported model: [GLocalKD, GLADC, SIGNET, GOOD-D, GraphDE, CVTGAD]."
                             "Default: GLADC")
    parser.add_argument("-gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("-data_root", default='data', type=str)

    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad','ood'])
    parser.add_argument('-DS', help='Dataset', default='DHFR') 
    parser.add_argument('-DS_ood', help='Dataset', default='ogbg-molsider')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-batch_size_test', type=int, default=64)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=2)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=2)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-n_train', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.3, help='Dropout rate.')

    
    subparsers = parser.add_subparsers()
    
    '''
    CVTGAD parameter
    '''
    CVTGAD_subparser = subparsers.add_parser('CVTGAD')
    CVTGAD_subparser.set_defaults(model='CVTGAD')
    CVTGAD_subparser.add_argument('-GNN_Encoder', type=str, default='GIN')  
    CVTGAD_subparser.add_argument('-graph_level_pool', type=str, default='global_mean_pool')
    
  
    '''
    GLADC parameter
    '''
    GLADC_subparser = subparsers.add_parser('GLADC')
    GLADC_subparser.set_defaults(model='GLADC')
    GLADC_subparser.add_argument('-max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    GLADC_subparser.add_argument('-output_dim', dest='output_dim', default=128, type=int, help='Output dimension')
    GLADC_subparser.add_argument('-nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    GLADC_subparser.add_argument('-nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')


    
    '''
    SIGNET parameter
    '''
    SIGNET_subparser = subparsers.add_parser('SIGNET')
    SIGNET_subparser.set_defaults(model='SIGNET')
    SIGNET_subparser.add_argument('--encoder_layers', type=int, default=5)
    SIGNET_subparser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
    SIGNET_subparser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
    SIGNET_subparser.add_argument('--explainer_model', type=str, default='gin', choices=['mlp', 'gin'])
    SIGNET_subparser.add_argument('--explainer_layers', type=int, default=5)
    SIGNET_subparser.add_argument('--explainer_hidden_dim', type=int, default=8)
    SIGNET_subparser.add_argument('--explainer_readout', type=str, default='add', choices=['concat', 'add', 'last'])

    '''
    GLocalKD
    '''
    GLocalKD_subparser = subparsers.add_parser('GLocalKD')
    GLocalKD_subparser.set_defaults(model='GLocalKD')
    GLocalKD_subparser.add_argument('-max-nodes', dest='max_nodes', type=int, default=0,
                        help='Maximum number of nodes (ignore graghs with nodes exceeding the number.')
    GLocalKD_subparser.add_argument('-clip', dest='clip', default=0.1, type=float, help='Gradient clipping.')
    GLocalKD_subparser.add_argument('-output_dim', dest='output_dim', default=256, type=int, help='Output dimension')
    GLocalKD_subparser.add_argument('-nobn', dest='bn', action='store_const', const=False, default=True,
                        help='Whether batch normalization is used')
    GLocalKD_subparser.add_argument('-nobias', dest='bias', action='store_const', const=False, default=True,
                        help='Whether to add bias. Default to True.')


    '''
    OCGTL
    '''
    OCGTL_subparser = subparsers.add_parser('OCGTL')
    OCGTL_subparser.set_defaults(model='OCGTL')
    OCGTL_subparser.add_argument('-norm_layer', default='gn')
    OCGTL_subparser.add_argument('-aggregation', default='add')
    OCGTL_subparser.add_argument('-bias', default=False)
    OCGTL_subparser.add_argument('-num_trans', default=6)

    '''
    OCGIN
    '''
    OCGIN_subparser = subparsers.add_parser('OCGIN')
    OCGIN_subparser.set_defaults(model='OCGIN')
    OCGIN_subparser.add_argument('-norm_layer', default='gn')
    OCGIN_subparser.add_argument('-aggregation', default='add')
    OCGIN_subparser.add_argument('-bias', default=False)

    '''
    GraphCL_IF
    '''
    GraphCL_IF_subparser = subparsers.add_parser('GraphCL_IF')
    GraphCL_IF_subparser.set_defaults(model='GraphCL_IF')
    GraphCL_IF_subparser.add_argument('-detector', default='IF')
    GraphCL_IF_subparser.add_argument('-IF_n_trees', type=int, default=200)
    GraphCL_IF_subparser.add_argument('-IF_sample_ratio', type=float, default=0.5)
    GraphCL_IF_subparser.add_argument('-gamma', default='scale')
    GraphCL_IF_subparser.add_argument('-nuOCSVM', type=float, default=0.1)

    '''
    GraphCL_OCSVM
    '''
    GraphCL_OCSVM_subparser = subparsers.add_parser('GraphCL_OCSVM')
    GraphCL_OCSVM_subparser.set_defaults(model='GraphCL_OCSVM')
    GraphCL_OCSVM_subparser.add_argument('-detector', default='OCSVM')
    GraphCL_OCSVM_subparser.add_argument('-IF_n_trees', type=int, default=200)
    GraphCL_OCSVM_subparser.add_argument('-IF_sample_ratio', type=float, default=0.5)
    GraphCL_OCSVM_subparser.add_argument('-gamma', default='scale')
    GraphCL_OCSVM_subparser.add_argument('-nuOCSVM', type=float, default=0.1)

    '''
    GraphCL_IF
    '''
    InfoGraph_IF_subparser = subparsers.add_parser('InfoGraph_IF')
    InfoGraph_IF_subparser.set_defaults(model='InfoGraph_IF')
    InfoGraph_IF_subparser.add_argument('-detector', default='IF')
    InfoGraph_IF_subparser.add_argument('-IF_n_trees', type=int, default=200)
    InfoGraph_IF_subparser.add_argument('-IF_sample_ratio', type=float, default=0.5)
    InfoGraph_IF_subparser.add_argument('-gamma', default='scale')
    InfoGraph_IF_subparser.add_argument('-nuOCSVM', type=float, default=0.1)

    '''
    GraphCL_OCSVM
    '''
    InfoGraph_OCSVM_subparser = subparsers.add_parser('InfoGraph_OCSVM')
    InfoGraph_OCSVM_subparser.set_defaults(model='InfoGraph_OCSVM')
    InfoGraph_OCSVM_subparser.add_argument('-detector', default='OCSVM')
    InfoGraph_OCSVM_subparser.add_argument('-IF_n_trees', type=int, default=200)
    InfoGraph_OCSVM_subparser.add_argument('-IF_sample_ratio', type=float, default=0.5)
    InfoGraph_OCSVM_subparser.add_argument('-gamma', default='scale')
    InfoGraph_OCSVM_subparser.add_argument('-nuOCSVM', type=float, default=0.1)

    '''
    KernelGLAD
    '''
    KernelGLAD_subparser = subparsers.add_parser('KernelGLAD')
    KernelGLAD_subparser.set_defaults(model='KernelGLAD')
    KernelGLAD_subparser.add_argument('-kernel', default='WL')
    KernelGLAD_subparser.add_argument('-detector', default='OCSVM')
    KernelGLAD_subparser.add_argument('-WL_iter', type=int, default=5)
    KernelGLAD_subparser.add_argument('-PK_bin_width', type=int, default=1)
    KernelGLAD_subparser.add_argument('-IF_n_trees', type=int, default=200)
    KernelGLAD_subparser.add_argument('-IF_sample_ratio', type=float, default=0.5)
    KernelGLAD_subparser.add_argument('-LOF_n_neighbors', type=int, default=20)
    KernelGLAD_subparser.add_argument('-LOF_n_leaf', type=int, default=30)
    KernelGLAD_subparser.add_argument('-detectorskernel', default='precomputed')
    KernelGLAD_subparser.add_argument('-nuOCSVM', type=float, default=0.1)



    args = parser.parse_args()

    main(args)
