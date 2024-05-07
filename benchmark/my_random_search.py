import tqdm
import torch
import argparse
import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
import warnings
from GAOOD.metric import *
from GAOOD.utils import load_data
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

    if args.exp_type == 'oodd':
        #datasets = ['AIDS+DHFR', 'ogbg-molbbbp+ogbg-molbace', 'BZR+COX2', 'ogbg-molclintox+ogbg-mollipo',
                    #'ENZYMES+PROTEINS', 'ogbg-molfreesolv+ogbg-moltoxcast', 'IMDB-MULTI+IMDB-BINARY', 'PTC_MR+MUTAG',
                    #'ogbg-molesol+ogbg-molmuv', 'ogbg-moltox21+ogbg-molsider']
        datasets = ['AIDS+DHFR', 'BZR+COX2']
    elif args.exp_type == 'ood':
        datasets=['GOOD','DrugOOD']
    for dataset_name in datasets:
        for metric in ['AUROC', 'AUPRC']:
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
            best_val_score = 0
            for t in tqdm.tqdm(range(num_trial)):

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
                    model.fit(dataset=None, args=args, label=None, dataloader=dataloader, dataloader_Val=dataloader_test)
                    if model.max_AUC > best_val_score:
                        print("****current best score****")
                        best_val_score = model.max_AUC
                        best_model_config = deepcopy(model_config)
                        score,pred = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args,return_score=False)
                        best_troc, best_tprc, = ood_auc(pred,score), ood_aupr(pred,score)
                else:
                    model.fit(dataset_train)
            best_model_configs[model][args.DS] = deepcopy(best_model_config)
            model_result[dataset_name + '-AUROC'] = best_troc
            model_result[dataset_name + '-AUPRC'] = best_tprc

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
    parser.add_argument("--model", type=str, default="dominant",
                        help="supported model: [lof, if, mlpae, scan, radar, "
                             "anomalous, gcnae, dominant, done, adone, "
                             "anomalydae, gaan, guide, conad]. "
                             "Default: dominant")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")
    parser.add_argument("--dataset", type=str, default='inj_cora',
                        help="supported dataset: [inj_cora, inj_amazon, "
                             "inj_flickr, weibo, reddit, disney, books, "
                             "enron]. Default: inj_cora")
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad'])
    parser.add_argument('-DS', help='Dataset', default='BZR')
    parser.add_argument('-DS_ood', help='Dataset', default='COX2')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=2)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.5)
    args = parser.parse_args()

    # global setting
    num_trial = 20

    main(args)
