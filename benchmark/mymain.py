import tqdm
import torch
import argparse
import warnings
import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
from GAOOD.metric import *
from utils import init_model
from dataloader.data_loader import *
'''
python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0
oodd（两个数据集OOD），ood:是GOOD/Drugood，ad :异常检测（tox/TU）
model：模型名字
DS_pair BZR+COX2  对应两个数据集的OOD
Ds, 剩下两个，ood和ad
超参

'''
def main(args):
    auc, ap, rec = [], [], []
    print(args)
    for _ in tqdm.tqdm(range(args.num_trial)):
        if args.exp_type == 'ad' and args.DS.startswith('Tox21'): 
                dataset_train,dataset_test,dataloader, dataloader_test, meta = get_ad_dataset_Tox21(args)
            # else:
            #     splits = get_ad_split_TU(args, fold=args.num_trial)
        elif args.exp_type == 'oodd':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_test, dataloader, dataloader_test, meta = get_ood_dataset(args)
            
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            splits = get_ad_split_TU(args, fold=args.num_trial)
            dataset_train, dataset_test, dataloader, dataloader_test, meta = get_ad_dataset_TU(args, splits[_])
            
        elif args.exp_type == 'ood':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_test, dataloader, dataloader_test, meta = get_ood_dataset_spilt(args)

        args.max_nodes_num = meta['max_nodes_num']
        args.dataset_num_features = meta['num_feat']
        args.n_train =  meta['num_train']
        args.n_edge_feat = meta['num_edge_feat']
        
        # args.dataset_num_features = meta['num_feat']
        # args.n_train =  meta['num_train']
        # args.max_nodes_num = meta['max_nodes_num']

        model = init_model(args)
        ###如果要自定义dataloader,就把dataset传进去，dataloader=None,否则按下面的来即可
        
        if args.model == 'GOOD-D':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader)
        elif args.model == 'GraphDE':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader)
        elif args.model == 'GLocalKD':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader)
        else:
            model.fit(dataset_train)

        score, y_all = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args, return_score=False)


        # print(score)
        # print(y_all)

        
        auc.append(ood_auc(y_all,score))
        print(auc)
        #auc.append(eval_roc_auc(y, score))
        #ap.append(eval_average_precision(y, score))
        #rec.append(eval_recall_at_k(y, score, k))

    auc = torch.tensor(auc)
    #ap = torch.tensor(ap)
    #rec = torch.tensor(rec)
    print(auc)
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
    parser.add_argument("--model", type=str, default="GLocalKD",
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
    
    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad','ood'])
    parser.add_argument('-DS', help='Dataset', default='BZR') 
    #BZR, DHFR
    #(BZR, COX2), (ogbg-moltox21,ogbg-molsider)
    #Tox21_PPAR-gamma
    parser.add_argument('-DS_ood', help='Dataset', default='ogbg-molsider')
    parser.add_argument('-DS_pair', default=None)
    parser.add_argument('-rw_dim', type=int, default=16)
    parser.add_argument('-dg_dim', type=int, default=16)
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-n_train', type=int, default=10)
    args = parser.parse_args()

    # global setting
    num_trial = 5

    main(args)
