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
    
def process_model_results(auc, ap, args):
    auc_final = sum(auc) / len(auc)
    ap_final = sum(ap) / len(ap)
    auc_variance = statistics.variance(auc)
    ap_variance = statistics.variance(ap)

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
    columns = ['name']
    model_result = {'name': args.model}
    results = pd.DataFrame(columns=columns)
    set_seed()
    for _ in tqdm.tqdm(range(args.num_trial)):

        if args.exp_type == 'ad':
            if args.DS.startswith('Tox21'):
                dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_Tox21(
                    args)
                for metric in ['AUROC', 'AUPRC']:
                    columns.append(args.DS + '-' + metric)
            else:
                splits = get_ad_split_TU(args, fold=args.num_trial)
        if args.exp_type == 'oodd':
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset(
                args)
            for metric in ['AUROC', 'AUPRC']:
                columns.append(args.DS_pair + '-' + metric)
        elif args.exp_type == 'ad' and not args.DS.startswith('Tox21'):
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ad_dataset_TU(
                args, splits[_])
            for metric in ['AUROC', 'AUPRC']:
                columns.append(args.DS + '-' + metric)
        elif args.exp_type == 'ood':
            print("-------")
            print(args.exp_type)
            dataset_train, dataset_val, dataset_test, dataloader, dataloader_val, dataloader_test, meta = get_ood_dataset_spilt(
                args)
            for metric in ['AUROC', 'AUPRC']:
                columns.append(args.DS + '-' + metric)

        

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

        auc.append(ood_auc(y_all, score))
        ap.append(ood_aupr(y_all, score))
        print("AUROC:", auc[-1])
        print("AUPRC:", ap[-1])
        
    process_model_results(auc, ap, args)

        # 计算平均值和方差
#     auc_final = sum(auc) / len(auc)
#     ap_final = sum(ap) / len(ap)
#     auc_variance = statistics.variance(auc)
#     ap_variance = statistics.variance(ap)

#     # 创建或更新结果DataFrame
#     model_result = {}
#     if args.exp_type == 'oodd':
#         file_id = args.model
#         # file_id = args.DS_pair + args.model
#         model_result[args.DS_pair + '-AUROC'] = f"{auc_final * 100:.2f}%"
#         model_result[args.DS_pair + '-AUROC_Var'] = f"{auc_variance * 100:.2f}%"
#         model_result[args.DS_pair + '-AUPRC'] = f"{ap_final * 100:.2f}%"
#         model_result[args.DS_pair + '-AUPRC_Var'] = f"{ap_variance * 100:.2f}%"
#     else:
#         # file_id = args.DS + args.model
#         file_id = args.model
#         model_result[args.DS + '-AUROC'] = f"{auc_final * 100:.2f}%"
#         model_result[args.DS + '-AUROC_Var'] = f"{auc_variance * 100:.2f}%"
#         model_result[args.DS + '-AUPRC'] = f"{ap_final * 100:.2f}%"
#         model_result[args.DS + '-AUPRC_Var'] = f"{ap_variance * 100:.2f}%"

#     model_result_df = pd.DataFrame([model_result])
#     results = pd.concat([results, model_result_df])
#     file_id = save_results_csv(results, file_id)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="GOOD-D",
                        help="supported model: [GLocalKD, GLADC, SIGNET, GOOD-D, GraphDE]."
                             "Default: GLADC")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU Index. Default: -1, using CPU.")

    parser.add_argument('-exp_type', type=str, default='ad', choices=['oodd', 'ad','ood'])
    parser.add_argument('-DS', help='Dataset', default='DHFR') 
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
    parser.add_argument('-num_trial', type=int, default=10)
    parser.add_argument('-num_epoch', type=int, default=150)
    parser.add_argument('-eval_freq', type=int, default=10)
    parser.add_argument('-is_adaptive', type=int, default=1)
    parser.add_argument('-num_cluster', type=int, default=2)
    parser.add_argument('-alpha', type=float, default=0)
    parser.add_argument('-n_train', type=int, default=10)
    parser.add_argument('-dropout', type=float, default=0.3, help='Dropout rate.')


    
    args = parser.parse_args()

    # 根据模型参数添加模型特有的参数
    if args.model == "GLADC":
        parser.add_argument('--max-nodes', type=int, default=0, help='Maximum number of nodes.')
        parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
        parser.add_argument('--output-dim', type=int, default=128, help='Output dimension.')
    elif args.model == "GLocalKD":
        parser.add_argument('--max-nodes', type=int, default=0, help='Maximum number of nodes.')
        parser.add_argument('--clip', type=float, default=0.1, help='Gradient clipping.')
        parser.add_argument('--batch-size', type=int, default=300, help='Batch size.')
        parser.add_argument('--hidden-dim', type=int, default=512, help='Hidden dimension.')
        parser.add_argument('--output-dim', type=int, default=256, help='Output dimension.')
        parser.add_argument('--num-gc-layers', type=int, default=3, help='Number of graph convolution layers.')
        parser.add_argument('--nobn', action='store_const', const=False, default=True, help='Whether batch normalization is used')
        parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate.')
        parser.add_argument('--nobias', action='store_const', const=False, default=True, help='Whether to add bias.')
    elif args.model == "SIGNET":
        parser.add_argument('--encoder_layers', type=int, default=5)
        parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max'])
        parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])
        parser.add_argument('--explainer_model', type=str, default='gin', choices=['mlp', 'gin'])
        parser.add_argument('--explainer_layers', type=int, default=5)
        parser.add_argument('--explainer_hidden_dim', type=int, default=8)
        parser.add_argument('--explainer_readout', type=str, default='add', choices=['concat', 'add', 'last'])

    # 解析新增加的参数
    args = parser.parse_args()

    main(args)
