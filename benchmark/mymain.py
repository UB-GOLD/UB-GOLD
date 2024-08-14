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
import pandas as pd
import statistics
import seaborn as sns
'''
python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0
oodd:inter datasets OOD,ood:intra dataset OOD,ad :anomaly detection（tox/TU）
model：name of model
DS_pair: parameter of oodd, such as :BZR+COX2  
Ds : dataset parameter for ood and ad



'''



def save_results_csv(model_result, model_name):
    # folder and name
    results_dir = 'results'
    filename = f'{results_dir}/{model_name}.csv'
    
    
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    # dictionary to DataFrame
    df = pd.DataFrame([model_result])
    
    
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, mode='w', header=True, index=False)

    print(f'Saved results to {filename}')
    
def process_model_results(auc, ap, rec, args):
    auc_final = sum(auc) / len(auc)
    ap_final = sum(ap) / len(ap)
    rec_final = sum(rec) / len(rec)
    auc_variance = statistics.variance(auc)
    ap_variance = statistics.variance(ap)
    rec_variance = statistics.variance(rec)

    model_result = {}
    file_id = args.model  
    
    
    if args.exp_type == 'oodd':
        key_prefix = args.DS_pair
    else:
        key_prefix = args.DS
    
    
    model_result['Dataset'] = key_prefix
    model_result['AUROC'] = f"{auc_final * 100:.2f}%"
    model_result['AUROC_Var'] = f"{auc_variance * 100:.2f}%"
    model_result['AUPRC'] = f"{ap_final * 100:.2f}%"
    model_result['AUPRC_Var'] = f"{ap_variance * 100:.2f}%"
    model_result['FPR95'] = f"{rec_final * 100:.2f}%"
    model_result['FPR95_Var'] = f"{rec_variance * 100:.2f}%"

    save_results_csv(model_result, file_id)


# def plot_distribution(vi_pos, vi_neg, filename, args):
#     """plot vi score distribution figure"""
#     sns.set(style='white')
#     fig,ax = plt.subplots(1,1,figsize=(4.5,3))

#     # plot vi score distribution without ground truth
#     sns.distplot(vi_pos, hist=False, ax=ax, kde_kws={'fill': True}, color='#7F95D1', label='in-distribution')
#     sns.distplot(vi_neg, hist=False, ax=ax, kde_kws={'fill': True}, color='#FF82A9', label='out-of-distribution')
#     ax.spines['bottom'].set_linewidth(0.5)
#     ax.spines['left'].set_linewidth(0.5)
#     ax.spines['top'].set_linewidth(0.5)
#     ax.spines['right'].set_linewidth(0.5)
#     ax.set_xlabel('OOD judge score', size=15)
#     ax.set_ylabel('Frequency', size=15)
    
#     title = f'{args.model} - {args.DS}'
#     # title = f'{args.model} - AI→DH'
#     ax.set_title(title, size=15)

#     handles, labels = ax.get_legend_handles_labels()
#     # fig.legend(handles, labels, loc='upper center', fontsize=10,ncol=2, bbox_to_anchor=(0.55, 1.08))
#     fig.legend(handles, labels, loc='upper right', fontsize=8, ncol=1, bbox_to_anchor=(0.95, 0.85))
   

#     fig.tight_layout()
#     fig.savefig(filename, bbox_inches='tight')
    
# def plot_score(score_iid, score_ood, exp_dir, args):
    
#     score_iid = np.array(score_iid)
#     score_ood = np.array(score_ood)

#     # 标准化处理
#     from sklearn.preprocessing import StandardScaler
#     scaler = StandardScaler()
#     score_iid = scaler.fit_transform(score_iid.reshape(-1, 1)).flatten()
#     score_ood = scaler.transform(score_ood.reshape(-1, 1)).flatten()
   
#     plot_distribution(score_iid, score_ood, os.path.join(exp_dir,f"{args.DS}-{args.model}.pdf"), args)
#     return

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
    auc, ap, rec = [], [], []
    model_result = {'name': args.model}
    
    # set_seed()
    # import ipdb
    # ipdb.set_trace()
    seed = 3407
    for i in tqdm.tqdm(range(args.num_trial)):
        set_seed(seed+i)
        if args.exp_type == 'ad':
            print("-------")
            print(args.exp_type)
            if args.DS.startswith('Tox21'):
                dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_Tox21(args)
            else:
                splits = get_ad_split_TU(args, fold=args.num_trial)
        if args.exp_type == 'oodd':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset(args)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_TU(args, splits[i])
        elif args.exp_type == 'ood':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset_spilt(args)
            

        

        args.max_nodes_num = meta['max_nodes_num']
        args.dataset_num_features = meta['num_feat']
        args.n_train =  meta['num_train']
        args.n_edge_feat = meta['num_edge_feat']

        model = init_model(args)
        ###If you want to define your own dataloader, just pass in the dataset, dataloader=None, otherwise press the following
        
        if args.model == 'GOOD-D':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'GraphDE':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'GLocalKD':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'GLADC':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'SIGNET':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'CVTGAD':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == "OCGTL":
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == "OCGIN":
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'GraphCL_IF':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'GraphCL_OCSVM':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'InfoGraph_IF':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'InfoGraph_OCSVM':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        elif args.model == 'KernelGLAD':
            print(args.model)
            model.fit(dataset=dataset_train, args=args, label=None, dataloader=dataloader, dataloader_val=dataloader_val)
        else:
            model.fit(dataset_train)

        score, y_all = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args, return_score=False)

        # exp_dir = "results"
        # score = np.array(score)
        # y_all = np.array(y_all)
        # score_iid = score[y_all == 0]
        # score_ood = score[y_all != 0]
        # plot_score(score_iid, score_ood, exp_dir, args)
        
        rec.append(fpr95(y_all, score))
        auc.append(ood_auc(y_all, score))
        ap.append(ood_aupr(y_all, score))
        print("AUROC:", auc[-1])
        print("AUPRC:", ap[-1])
        print("FPR95:", rec[-1])
        
    process_model_results(auc, ap, rec, args)


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
    parser.add_argument('-batch_size', type=int, default=128)
    parser.add_argument('-batch_size_test', type=int, default=9999)
    parser.add_argument('-lr', type=float, default=0.0001)
    parser.add_argument('-num_layer', type=int, default=5)
    parser.add_argument('-hidden_dim', type=int, default=16)
    parser.add_argument('-num_trial', type=int, default=5)
    parser.add_argument('-num_epoch', type=int, default=400)
    parser.add_argument('-eval_freq', type=int, default=4)
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
