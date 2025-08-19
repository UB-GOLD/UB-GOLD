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
from .drugood_dataset import DrugOODDataset
from torch.utils.data import ConcatDataset
# from GOOD import register

def read_graph_file(DS, path):
    if "training" in DS:
        path = path+"_training/"
    else:
        path = path+"_testing/"
    # print(path)
    prefix = os.path.join(path, DS)
  
    filename_graph_indic = prefix + '_graph_indicator.txt'
    graph_indic = {}

    with open(filename_graph_indic) as f:
      
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

def split_list(data):
    random.shuffle(data)  
    n = len(data)
    n1 = int(0.7 * n)  
    n2 = int(0.1 * n)  
    n3 = n2  
    n4 = n - n1 - n2 - n3 

    list1 = data[:n1]
    list2 = data[n1:n1 + n2]
    list3 = data[n1 + n2:n1 + n2 + n3]
    list4 = data[n1 + n2 + n3:]
    return list1, list2, list3, list4

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


# DS dataset
def get_ood_dataset(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    path = osp.join(path_now, '.', 'data', DS)
    path_ood = osp.join(path_now, '.', 'data', DS_ood)
    
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

    # num_sample = len(dataset)
    # num_train = int(num_sample * train_per)
    # indices = torch.randperm(num_sample)
    # idx_train = torch.sort(indices[:num_train])[0]
    # idx_val = torch.sort(indices[num_train:])[0]
    # idx_test = torch.sort(indices[num_train:])[0]

    # dataset_train = dataset[idx_train]
    # dataset_val = dataset[idx_val]
    # dataset_test = dataset[idx_test]
    # dataset_ood = dataset_ood[: len(dataset_test)]

    num_sample = len(dataset)
    num_train = int(num_sample * train_per)
    indices = torch.randperm(num_sample)
    idx_train = torch.sort(indices[:num_train])[0]
    idx_test = torch.sort(indices[num_train:])[0]

    dataset_train = dataset[idx_train]
    dataset_test = dataset[idx_test]
    dataset_ood = dataset_ood[: len(dataset_test)]

    data_list_train = []
    # data_list_val = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_train = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

  

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
    dataset_val = dataset_test
    dataloader_val = dataloader_test
    meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
            'num_test':len(dataset_test), 'num_ood':len(dataset_ood),'max_nodes_num':max_nodes_num,'num_edge_feat':0}

    #train (ID), test (ID+OOD), train dataloader, test dataloader, meta
    return dataset_train, dataset_val, dataset_test, dataloader_train, dataloader_val, dataloader_test, meta

# # DS for GOOD, DrugOOD
# def get_ood_dataset_spilt(args, train_per=0.9, need_str_enc=True):
#     if args.DS_pair is not None:
#         DSS = args.DS_pair.split("+")
#         DS, DS_ood = DSS[0], DSS[1]
#     else:
#         DS, DS_ood = args.DS, args.DS_ood

#     DrugooD = DS.startswith('DrugOOD')
 
#     path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
 
#     path = osp.join(path_now, '.', 'data', DS)

#     n_train_data, n_in_test_data, n_out_test_data = 1000, 500, 500
#     if DrugooD:
#         is_drug,DS_drug = DS.split("+")

#         print(DS_drug)
#         dataset_all = DrugOODDataset(name = DS_drug, root = args.data_root)
        
#         random.shuffle(dataset_all.train_index)
#         random.shuffle(dataset_all.test_index)
        
#         dataset = dataset_all[dataset_all.train_index]
#         dataset_ood = dataset_all[dataset_all.test_index]
#         max_nodes_num = max([_.num_nodes for _ in dataset_all])
#         dataset.data.x = dataset.data.x.type(torch.float32)

#         # print(len(dataset_ood))
#         dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)    
       
#     else:
#         dataset_name,domain,shift = DS.split("+")
     
#         root = os.getcwd()+"/"+args.data_root

#         datasets, meta_info = register.datasets[dataset_name].load(dataset_root=args.data_root,
#                                                                               domain=domain,
#                                                                               shift=shift,
#                                                                               generate = False,
#                                                                               )
       
            
#         dataset = datasets["train"]
#         perm_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
#         dataset = dataset[perm_idx]
#         dataset.data.x = dataset.data.x.type(torch.float32)
#         dataset_ood = datasets["test"]
#         perm_idx = torch.randperm(len(dataset_ood), generator=torch.Generator().manual_seed(0))
#         dataset_ood  =dataset_ood[perm_idx]
#         max_nodes_num = max([_.num_nodes for _ in dataset])

#         dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)

#     max_nodes_num_train = max([_.num_nodes for _ in dataset])
#     max_nodes_num_test = max([_.num_nodes for _ in dataset_ood])
#     max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test
    
#     dataset_num_features = dataset.num_node_features
#     dataset_num_features_ood = dataset_ood.num_node_features
#     assert dataset_num_features == dataset_num_features_ood

#     # dataset_id = dataset[:n_train_data]
#     # dataset_train = dataset_id[:int(n_train_data*train_per)]
#     # dataset_val = dataset_id[int(n_train_data*train_per):]
#     # dataset_test = dataset[n_train_data:n_train_data + n_in_test_data]
#     # dataset_ood = dataset_ood[: len(dataset_test)]
    
#     dataset_train = dataset[:n_train_data]
#     dataset_test = dataset[n_train_data:n_train_data+n_in_test_data]
#     dataset_ood = dataset_ood[: len(dataset_test)]
    
#     data_list_train = []
#     # data_list_val = []
#     idx = 0
#     for data in dataset_train:
#         data.y = 0
#         data['idx'] = idx
#         idx += 1
#         data_list_train.append(data)

#     if need_str_enc:
#         data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
#     dataloader_train = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

#     data_list_test = []
#     for data in dataset_test:
#         data.y = 0
#         data.edge_attr = None
#         if not DrugooD:
#           data.env_id = data.domain_id
#         data_list_test.append(data)

#     for data in dataset_ood:
#         data.y = 1
#         data.edge_attr = None
#         if not DrugooD:
#           data.env_id = data.domain_id
#         data_list_test.append(data)

#     if need_str_enc:
#         data_list_test = init_structural_encoding(data_list_test, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
#     dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=False)
#     dataset_test = ConcatDataset([dataset_test, dataset_ood])
#     dataset_val = dataset_test
#     dataloader_val = dataloader_test
    
#     meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
#             'num_test':len(dataset_test), 'num_ood':len(dataset_ood),'max_nodes_num':max_nodes_num,'num_edge_feat':0}
#     print(meta)
#     #train (ID), test (ID+OOD), train dataloader, test dataloader, meta
#     return dataset_train, dataset_val, dataset_test, dataloader_train, dataloader_val, dataloader_test, meta




def get_ad_split_TU(args, fold=5):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    print(path_now)

    path = osp.join(path_now, '.', 'data', args.DS)

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

#10 TU dataset
def get_ad_dataset_TU(args, split, need_str_enc=True):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
   
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
    
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
 
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    #train (ID), test (ID+OOD), train dataloader, test dataloader, meta

    dataloader_val = dataloader_test
    return dataset[train_index], dataset[test_index],dataset[test_index], dataloader_train, dataloader_val, dataloader_test, meta

#4 real dataset
def get_ad_dataset_Tox21(args, need_str_enc=True):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))

    path = osp.join(path_now, '.', 'data', args.DS)

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
        # data_val  = init_structural_encoding(data_val, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)

    data_val = data_test
    dataloader_val = dataloader_test
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    #train (ID), test (ID+OOD), train dataloader, test dataloader, meta
    return data_train,data_val, data_test, dataloader_train,dataloader_val, dataloader_test, meta


def perturbation_datasets(args, split, need_str_enc=True):
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
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
            

    data_id=[]
    data_ood=[]
    for data in data_test:
        # data.y = 1 if data.y == 0 else 0
        if data.y == 0:
            data.y = 1
            data_ood.append(data)
        else:
            data.y = 0
            data_id.append(data)

    data_ood, per_1, per_2, per_3 = split_list(data_ood)
    
    data_test = data_ood+data_id
    
    if args.per==0.1:
        data_train = data_train+per_1
    elif args.per==0.2:
        data_train = data_train+per_1+per_2
    else:
        data_train = data_train+per_1+per_2+per_3
    
    idx = 0
    for data in data_train:
        data.y = 0
        data['idx'] = idx
        idx += 1

    
    
    
    
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(data_train), 'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    dataloader_val = dataloader_test
    return dataset[train_index], dataset[test_index],dataset[test_index], dataloader_train, dataloader_val, dataloader_test, meta

    
    
    
    
    
def get_ood_dataset_near_and_far(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    TU = not DS.startswith('ogbg-mol')
    path_now = os.path.abspath(os.path.join(os.getcwd(), "."))
    path = osp.join(path_now, '.', 'data', DS)
    path_ood = osp.join(path_now, '.', 'data', DS_ood)
    
    
    # dataset(ID,nearOOD), dataset_OOD(farOOD)
    if TU:
        dataset = TUDataset(path, name=DS, transform=(Constant(1, cat=False)))
        dataset_ood = TUDataset(path_ood, name=DS_ood, transform=(Constant(1, cat=False)))
    else:
        dataset = PygGraphPropPredDataset(name=DS, root=path)
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = (PygGraphPropPredDataset(name=DS_ood, root=path_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
    split = get_ad_split_TU(args)


    data_list = []
    label_list = []

    for data in dataset:
        data.edge_attr = None
        data_list.append(data)
        label_list.append(data.y.item())
    if need_str_enc:
        data_list = init_structural_encoding(data_list, rw_dim=args.rw_dim, dg_dim=args.dg_dim)


    (train_index, test_index) = split[0]
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
        
    data_near_ood = []
    for data in data_test:
        if data.y == 1:
            data_near_ood.append(data)
    # dataset_nearood = dataset[test_index]
   


    max_nodes_num_train = max([_.num_nodes for _ in dataset])
    max_nodes_num_test = max([_.num_nodes for _ in dataset_ood])
    max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test

    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood


    dataset_train = data_train
    dataset_test = data_test
    
    
    # data_list_test = []
    data_id = []
    for data in dataset_test:
        if data.y == 0:
            data.edge_attr = None
            # data_list_test.append(data)
            data_id.append(data)
        
    len_ood = len(data_near_ood) if len(data_near_ood) < len(data_id) else len(data_id)
    data_near_ood = data_near_ood[: len_ood]
    dataset_ood = dataset_ood[: len_ood]
    data_id = data_id[: len_ood]
    print(len_ood)
    # len(data_far_ood)
    # len(data_near_ood)

    data_list_train = []
    # data_list_val = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_train = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

        
    data_far_ood = []
    for data in dataset_ood:
        data.y = 1
        data.edge_attr = None
        # data_list_test.append(data)
        data_far_ood.append(data)
    # import ipdb
    # ipdb.set_trace()
    
    data_near_ood = data_near_ood+data_id
    data_far_ood = data_far_ood+data_id
    if need_str_enc:
        data_far_ood = init_structural_encoding(data_far_ood, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
        data_near_ood = init_structural_encoding(data_near_ood, rw_dim=args.rw_dim, dg_dim=args.dg_dim)

    dataloader_near_ood =  DataLoader(data_near_ood, batch_size=args.batch_size_test, shuffle=True)
    dataloader_far_ood =  DataLoader(data_far_ood, batch_size=args.batch_size_test, shuffle=True)
    # dataset_test = ConcatDataset([dataset_test, dataset_ood])
    # dataset_val = dataset_test
    # dataloader_val = dataloader_test
    
   
    meta = {'num_feat': dataset_num_features, 'num_train': len(dataset_train),
            'num_test': len(dataset_test), 'num_ood': len(data_near_ood)*2, 'max_nodes_num': max_nodes_num,
            'num_edge_feat': 0, 'num_near_ood':len(data_near_ood),'num_far_ood':len(data_near_ood)}
    #train (ID), test (ID+OOD), train dataloader, test dataloader, meta
    return  0,0,0, dataloader_train, dataloader_near_ood, dataloader_far_ood, meta


def get_ood_dataset_near_and_far_size(args, train_per=0.9, need_str_enc=True):
    if args.DS_pair is not None:
        DSS = args.DS_pair.split("+")
        DS, DS_ood = DSS[0], DSS[1]
    else:
        DS, DS_ood = args.DS, args.DS_ood

    DrugooD = DS.startswith('DrugOOD')
    path_now =  os.path.abspath(os.path.join(os.getcwd(), "."))
    path = osp.join(path_now, '.', 'data', DS)
    n_train_data, n_in_test_data, n_out_test_data = 1000, 500, 500
    if DrugooD:
        is_drug,DS_drug = DS.split("+")
        #dataset = DrugOOD(path, mode='iid')
        print(DS_drug)
        dataset_all = DrugOODDataset(name = DS_drug, root = args.data_root)
        
        random.shuffle(dataset_all.train_index)
        random.shuffle(dataset_all.test_index)
        
        dataset = dataset_all[dataset_all.train_index]
        dataset_ood = dataset_all[dataset_all.test_index]
        max_nodes_num = max([_.num_nodes for _ in dataset_all])
        dataset.data.x = dataset.data.x.type(torch.float32)

        # print(len(dataset_ood))
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)    
       
    else:
        dataset_name,domain,shift = DS.split("+")
        # print(os.getcwd())
        root = os.getcwd()+"/"+args.data_root
      
        datasets, meta_info = register.datasets[dataset_name].load(dataset_root=args.data_root,
                                                                              domain=domain,
                                                                              shift=shift,
                                                                              generate = False,
                                                                              )
       
            
        dataset = datasets["train"]
        perm_idx = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(0))
        dataset = dataset[perm_idx]
        
        dataset.data.x = dataset.data.x.type(torch.float32)
        dataset_ood = datasets["test"]
        

        size_id = []
        size_ood_near = []
        size_ood_far = []
        data_ood_far=[]
        data_ood_near=[]
        print()
        for _ in dataset[:1000]:
            size_id.append(_.num_nodes)
        print("avgid{}".format(sum(size_id)/len(size_id)))
        avb_id = int(sum(size_id)/len(size_id))
        data_ood_near = []
        data_ood_near_size = []
        for d in dataset_ood:
            if d.num_nodes in range(avb_id-5,avb_id+5):
                data_ood_near.append(d)
                data_ood_near_size.append(d.num_nodes)
            if len(data_ood_near) == 500:
                break
                
        print("avg__ood_near{}".format(sum(data_ood_near_size)/len(data_ood_near_size)))
        
        
        
        dataset_ood.data.x = dataset_ood.data.x.type(torch.float32)
        for d in dataset_ood[:500]:
            data_ood_far.append(d)
    
    
    dataset_num_features = dataset.num_node_features
    dataset_num_features_ood = dataset_ood.num_node_features
    assert dataset_num_features == dataset_num_features_ood

    # dataset_id = dataset[:n_train_data]
    # dataset_train = dataset_id[:int(n_train_data*train_per)]
    # dataset_val = dataset_id[int(n_train_data*train_per):]
    # dataset_test = dataset[n_train_data:n_train_data + n_in_test_data]
    # dataset_ood = dataset_ood[: len(dataset_test)]
    
    dataset_train = dataset[:n_train_data]
    dataset_test = dataset[n_train_data:n_train_data+n_in_test_data]

    dataset_ood = dataset_ood[: len(dataset_test)]
    max_nodes_num_train = max([_.num_nodes for _ in dataset])
    max_nodes_num_test = max([_.num_nodes for _ in dataset_ood])
    max_nodes_num = max_nodes_num_train if max_nodes_num_train > max_nodes_num_test else max_nodes_num_test
    print(max_nodes_num_train)
    data_list_train = []
    # data_list_val = []
    idx = 0
    for data in dataset_train:
        data.y = 0
        data['idx'] = idx
        idx += 1
        data_list_train.append(data)

    if need_str_enc:
        data_list_train = init_structural_encoding(data_list_train, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_train = DataLoader(data_list_train, batch_size=args.batch_size, shuffle=True)

    
    
    data_list_test = []
    for data in dataset_test:
        data.y = 0
        data.edge_attr = None
        if not DrugooD:
            data.env_id = data.domain_id
        data_list_test.append(data)
    data_id=data_list_test
    
    


    data_oodnear = []
    for data in data_ood_near:
        data.y = 1
        data.edge_attr = None
        if not DrugooD:
              data.env_id = data.domain_id
        data_oodnear.append(data)
  
    data_oodfar = []
    for data in data_ood_far:
        data.y = 1
        data.edge_attr = None
        if not DrugooD:
              data.env_id = data.domain_id
        data_oodfar.append(data)

    data_near_ood=data_oodnear+data_id
    data_far_ood=data_oodfar+data_id
    
    if need_str_enc:
        data_near_ood = init_structural_encoding(data_near_ood, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
        data_far_ood = init_structural_encoding(data_far_ood, rw_dim=args.rw_dim, dg_dim=args.dg_dim)
    dataloader_test = DataLoader(data_list_test, batch_size=args.batch_size_test, shuffle=False)
    dataset_test = ConcatDataset([dataset_test, dataset_ood])
    dataset_val = dataset_test
    dataloader_val = dataloader_test
    dataloader_near_ood =  DataLoader(data_near_ood, batch_size=args.batch_size_test, shuffle=True)
    dataloader_far_ood =  DataLoader(data_far_ood, batch_size=args.batch_size_test, shuffle=True)
    meta = {'num_feat':dataset_num_features, 'num_train':len(dataset_train),
            'num_test':len(dataset_test), 'num_ood':len(dataset_ood),'max_nodes_num':max_nodes_num,'num_edge_feat':0}
    print(meta)
    return  0,0,0, dataloader_train, dataloader_near_ood, dataloader_far_ood, meta
