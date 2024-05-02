import os
import re
import os.path as osp
from scipy import sparse as sp
import torch
import numpy as np
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Constant
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_scipy_sparse_matrix, degree, from_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from sklearn.model_selection import StratifiedKFold
from .good_hiv import GOODHIV
import random
from .drugood_dataset import DrugOOD
from torch.utils.data import ConcatDataset
def read_graph_file(DS, path):
    if "training" in DS:
        path = path+"_training/"
    else:
        path = path+"_testing/"
    # print(path)
    prefix = os.path.join(path, DS)
    # print(prefix)
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}
    # print(filename_graph_indic)
    with open(filename_graph_indic) as f:
        #print("sucess")
        #exit()
        i = 1
        for line in f:
            line = line.strip("\n")
            graph_indic[i] = int(line)
            i += 1

    filename_nodes = prefix + '_node_labels.txt'
    node_labels = []
    try:
        with open(filename_nodes) as f:
            for line in f:
                line = line.strip("\n")
                node_labels += [int(line) - 1]
        num_unique_node_labels = max(node_labels) + 1
    except IOError:
        print('No node labels')

    filename_node_attrs = prefix + '_node_attributes.txt'
    node_attrs = []
    try:
        with open(filename_node_attrs) as f:
            for line in f:
                line = line.strip("\s\n")
                attrs = [float(attr) for attr in re.split("[,\s]+", line) if not attr == '']
                node_attrs.append(np.array(attrs))
    except IOError:
        print('No node attributes')

    label_has_zero = False
    filename_graphs = prefix + '_graph_labels.txt'
    graph_labels = []

    label_vals = []
    with open(filename_graphs) as f:
        for line in f:
            line = line.strip("\n")
            val = int(line)
            if val not in label_vals:
                label_vals.append(val)
            graph_labels.append(val)

    label_map_to_int = {val: i for i, val in enumerate(label_vals)}
    graph_labels = np.array([label_map_to_int[l] for l in graph_labels])

    filename_adj = prefix + '_A.txt'
    adj_list = {i: [] for i in range(1, len(graph_labels) + 1)}
    index_graph = {i: [] for i in range(1, len(graph_labels) + 1)}
    num_edges = 0
    with open(filename_adj) as f:
        for line in f:
            line = line.strip("\n").split(",")
            e0, e1 = (int(line[0].strip(" ")), int(line[1].strip(" ")))
            adj_list[graph_indic[e0]].append((e0, e1))
            index_graph[graph_indic[e0]] += [e0, e1]
            num_edges += 1
    for k in index_graph.keys():
        index_graph[k] = [u - 1 for u in set(index_graph[k])]

    graphs = []
    for i in range(1, 1 + len(adj_list)):
        G = nx.from_edgelist(adj_list[i])
        G.graph['label'] = graph_labels[i - 1]

        mapping = {}
        it = 0
        for n in G.nodes:
            mapping[n] = it
            it += 1

        G_pyg = from_networkx(nx.relabel_nodes(G, mapping))
        G_pyg.y = G.graph['label']
        G_pyg.x = torch.ones((G_pyg.num_nodes,1))

        if G_pyg.num_nodes > 0:
            graphs.append(G_pyg)

    return graphs


def init_structural_encoding(gs, rw_dim=16, dg_dim=16):
    for g in gs:
        A = to_scipy_sparse_matrix(g.edge_index, num_nodes=g.num_nodes)
        D = (degree(g.edge_index[0], num_nodes=g.num_nodes) ** -1.0).numpy()

        Dinv = sp.diags(D)
        RW = A * Dinv
        M = RW

        RWSE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(rw_dim-1):
            M_power = M_power * M
            RWSE.append(torch.from_numpy(M_power.diagonal()).float())
        RWSE = torch.stack(RWSE,dim=-1)

        g_dg = (degree(g.edge_index[0], num_nodes=g.num_nodes)).numpy().clip(0, dg_dim - 1)
        DGSE = torch.zeros([g.num_nodes, dg_dim])
        for i in range(len(g_dg)):
            DGSE[i, int(g_dg[i])] = 1

        g['x_s'] = torch.cat([RWSE, DGSE], dim=1)

    return gs

#DS 是ID,DS_OOD是另一个数据集，一共两个
def get_ood_dataset(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    # print(path_now)
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    #path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)
    path = osp.join(path_now, '.', 'data', DS)
    path_ood = osp.join(path_now, '.', 'data', DS_ood)
    # print(path)
    #print(os.path.abspath(os.path.dirname(os.getcwd())))
    #print(os.path.abspath(os.path.join(os.getcwd(), ".")))
    # print(path_ood)
    
    if TU:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
        dataset_ood = TUDataset(path_ood, name=DS_ood, transform=(Constant(1, cat=False)))
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
    max_nodes_num_train = max([_.num_nodes for _ in dataset])
    max_nodes_num_test = max([_.num_nodes for _ in dataset_ood])
    max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train:])[0]

    dataset_train = dataset[idx_train]
    dataset_test = dataset[idx_test]
    dataset_ood = dataset_ood[: len(dataset_test)]

    data_list_train = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data.edge_attr = None
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        data.edge_attr = None
        data_list_test.append(data)

    if need_str_enc:
        data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=True)
    dataset_test = ConcatDataset([dataset_test, dataset_ood])
    meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
            'num_test':len(dataset_test), 'num_ood':len(dataset_ood),'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    #训练集（ID）， 测试集（ID+OOD）， 训练集dataloader,测试集dataloader,数据信息
    return dataset_train, dataset_test, dataloader, dataloader_test, meta

#GOOD和DrugOOD， 一个数据集
def get_ood_dataset_spilt(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    #TU = not DS.startswith('ogbg-mol')
    DrugooD = DS.startswith('DrugOOD')
    # print(DrugooD)
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    # print(path_now)
    path = osp.join(path_now, '.', 'data', DS)
    # print(path)
    n_train_data, n_val_data, n_in_test_data, n_out_test_data = 500, 500, 500, 500
    if DrugooD:
        dataset = DrugOOD(path, mode='iid')
        print(len(dataset))
        max_nodes_num = max([_.num_nodes for _ in dataset])
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = DrugOOD(path, mode='ood')
        print(len(dataset_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)    

    else:
        from .register import register
        register.datasets ={"GOODHIV": GOODHIV(root=osp.join(r"/home/wangyili/shenxu/G-OOD-D-main/data/", 'hiv/'),
                                                              domain="size",
                                                              shift="covariate")}
        
        # datasets, meta_info = register.datasets["GOODHIV"].load(dataset_root=osp.join(r"/home/wangyili/shenxu/G-OOD-D-main/data/", 'hiv/'),
        #                                                       domain="size",
        #                                                       shift="covariate",
        #                                                      )
        #
    #iid_dataset = dataset["train"]
    #ood_dataset = DrugOOD(osp.join(dataset_dir, 'DrugOOD/'), mode='ood')

    
    
        dataset = datasets["train"][:1000]
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = datasets["test"][:n_out_test_data]
        max_nodes_num = max([_.num_nodes for _ in dataset])
        
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
    '''
    if TU and DrugooD:
        dataset = TUDataset(r"/home/wangyili/shenxu/G-OOD-D-main/data", name=DS, transform=(Constant(1, cat=False)))
        dataset_ood = TUDataset(r"/home/wangyili/shenxu/G-OOD-D-main/data", name=DS_ood, transform=(Constant(1, cat=False)))
    else:
        if DS.startswith('DrugOOD'):
            #print(path)
            dataset = DrugOOD(path, mode='iid')
            print(len(dataset))
            dataset.data.x = dataset.data.x.type(torch.float32)
            dataset_ood = DrugOOD(path, mode='ood')
            print(len(dataset_ood))
            dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
        else:
            dataset = PygGraphPropPredDataset(name=DS, root=path)
            dataset.data.x = dataset.data.x.type(torch.float32)
            dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood))
            dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
    '''

    max_nodes_num_train = max([_.num_nodes for _ in dataset])
    max_nodes_num_test = max([_.num_nodes for _ in dataset_ood])
    max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test
    
    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood

    dataset_train = dataset[:n_train_data]
    dataset_test = dataset[n_train_data:n_train_data+n_in_test_data]
    dataset_ood = dataset_ood[: len(dataset_test)]
    
    data_list_train = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data.edge_attr = None
        if not DrugooD:
          data.env_id = data.domain_id
        data_list_test.append(data)

    for data in dataset_ood:
        data.y = 1
        data.edge_attr = None
        if not DrugooD:
          data.env_id = data.domain_id
        data_list_test.append(data)

    if need_str_enc:
        data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=False)
    dataset_test = ConcatDataset([dataset_test, dataset_ood])
    meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
            'num_test':len(dataset_test), 'num_ood':len(dataset_ood),'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    
    #训练集（ID）， 测试集（ID+OOD）， 训练集dataloader,测试集dataloader,数据信息
    return dataset_train,dataset_test,dataloader, dataloader_test, meta


def get_ad_split_TU(args, fold=5):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    print(path_now)
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    #path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)
    path = osp.join(path_now, '.', 'data', args.DS)
    #path_ood = osp.join(path_now, '.', 'data', DS_ood)
    #DS = args.DS
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)

    dataset = TUDataset(path, name=args.DS)
    data_list = []
    label_list = []

    for data in dataset:
        data_list.append(data)
        label_list.append(data.y.item())

    kfd = StratifiedKFold(n_splits=fold, random_state=0, shuffle=True)

    splits = []
    for k, (train_index, test_index) in enumerate(kfd.split(data_list, label_list)):
        splits.append((train_index, test_index))

    return splits

#TU 10个数据集
def get_ad_dataset_TU(args, split, need_str_enc=True):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    # print(path_now)
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    #path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)
    path = osp.join(path_now, '.', 'data', args.DS)

    if args.DS in ['IMDB-BINARY', 'REDDIT-BINARY', 'COLLAB']:
        dataset = TUDataset(path, name=args.DS, transform=(Constant(1, cat=False)))
    else:
        dataset = TUDataset(path, name=args.DS)

    dataset_num_features = dataset.num_node_features
    max_nodes_num = max([_.num_nodes for _ in dataset])

    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())

    if need_str_enc:
        data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    (train_index, test_index) = split
    data_train_ = [data_list[i] for i in train_index]
    data_test = [data_list[i] for i in test_index]

    data_train = []
    for data in data_train_:
        if data.y != 0:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 0 else 0
    
    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    #训练集（ID）， 测试集（ID+OOD）， 训练集dataloader,测试集dataloader,数据信息
    return dataset[train_index],dataset[test_index],dataloader, dataloader_test, meta

#4个真实数据集
def get_ad_dataset_Tox21(args, need_str_enc=True):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    # print(path_now)
    #path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS)
    #path_ood = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', DS_ood)
    path = osp.join(path_now, '.', 'data', args.DS)
    # import ipdb; ipdb.set_trace()
    data_train_ = read_graph_file(args.DS + '_training', path)
    
    data_test = read_graph_file(args.DS + '_testing', path)
    
    dataset_num_features = data_train_[0].num_features

    max_nodes_num_train = max([_.num_nodes for _ in data_train_])
    max_nodes_num_test = max([_.num_nodes for _ in data_test])
    max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test
    
    data_train = []
    for data in data_train_:
        if data.y == 1:
            data_train.append(data)

    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    for data in data_test:
        data.y = 1 if data.y == 1 else 0

    if need_str_enc:
        data_train = init_structural_encoding(data_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
        data_test = init_structural_encoding(data_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    dataloader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    #训练集（ID）， 测试集（ID+OOD）， 训练集dataloader,测试集dataloader,数据信息
    return data_train,data_test,dataloader, dataloader_test, meta
