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
'''
python benchmark/mymain.py -exp_type oodd -DS_pair BZR+COX2 -num_epoch 400 -num_cluster 2 -alpha 0
oodd（两个数据集OOD），ood:是GOOD/Drugood，ad :异常检测（tox/TU）
model：模型名字
DS_pair BZR+COX2  对应两个数据集的OOD
Ds, 剩下两个，ood和ad
超参

'''
def save_results_csv(model_result, model_name):
    # 指定结果文件夹和文件名
    results_dir = 'results'
    filename = f'{results_dir}/{model_name}.csv'
    
    # 确保结果文件夹存在
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    # 将字典转换为DataFrame
    df = pd.DataFrame([model_result])
    
    # 检查文件是否存在来决定是否写入表头
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
    file_id = args.model  # 使用模型名称作为文件标识
    
    # 根据不同的实验类型添加数据
    if args.exp_type == 'oodd':
        key_prefix = args.DS_pair
    else:
        key_prefix = args.DS
    
    # 格式化数据并添加到结果字典中
    model_result['Dataset'] = key_prefix
    model_result['AUROC'] = f"{auc_final * 100:.2f}%"
    model_result['AUROC_Var'] = f"{auc_variance * 100:.2f}%"
    model_result['AUPRC'] = f"{ap_final * 100:.2f}%"
    model_result['AUPRC_Var'] = f"{ap_variance * 100:.2f}%"
    model_result['FPR95'] = f"{rec_final * 100:.2f}%"
    model_result['FPR95_Var'] = f"{rec_variance * 100:.2f}%"

    save_results_csv(model_result, file_id)




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
    
    set_seed()
    # import ipdb
    # ipdb.set_trace()
    
    for _ in tqdm.tqdm(range(args.num_trial)):

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
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_TU(args, splits[_])
        elif args.exp_type == 'ood':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset_spilt(args)
            

        

        args.max_nodes_num = meta['max_nodes_num']
        args.dataset_num_features = meta['num_feat']
        args.n_train =  meta['num_train']
        args.n_edge_feat = meta['num_edge_feat']

        model = init_model(args)
        ###如果要自定义dataloader,就把dataset传进去，dataloader=None,否则按下面的来即可
        
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
        else:
            model.fit(dataset_train)

        score, y_all = model.predict(dataset=dataset_test, dataloader=dataloader_test, args=args, return_score=False)
        
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



    args = parser.parse_args()

    main(args)
