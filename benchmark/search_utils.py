import random

import os
import json
import numpy as np
from GAOOD.detector import *


def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id


model_detector_dict = {
    # Classic Methods
    'GOOD-D': GOOD_D,
    'GraphDE': GraphDE,
    'GLocalKD':GLocalKD,
    "GLADC":GLADC,
    "SIGNET":SIGNET, 
    "CVTGAD":CVTGAD, 
    "OCGTL":OCGTL,
    "OCGIN":OCGIN,
    "GraphCL_IF":GraphCL_IF,
    "GraphCL_OCSVM":GraphCL_OCSVM,
    "InfoGraph_IF":InfoGraph_IF,
    "InfoGraph_OCSVM":InfoGraph_OCSVM,
    "KernelGLAD":KernelGLAD,
    
}


def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)

    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT'] and dataset in ['tfinance', 'dgraphfin', 'tsocial']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    if dataset in ['dgraphfin', 'tsocial']:
        if 'k' in model_config:
            model_config['k'] = min(5, model_config['k'])
        if 'num_cluster' in model_config:
            model_config['num_cluster'] = 2
        # if 'num_layers' in model_config:
        #     model_config['num_layers'] = min(2, model_config['num_layers'])
    return model_config


param_space = {}

param_space['GOOD-D'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GraphDE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['CVTGAD'] = {
    'h_feats': [16, 32, 64,128],
    'num_layers': [1, 2, 3,4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
    'GNN_Encoder': ['GIN','GAT','GCN'],
    'graph_level_pool':['global_mean_pool','global_max_pool']
}

param_space['GLADC'] = {
    'h_feats': [16, 32, 64,128],
    'num_layers': [1, 2, 3,4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
    'nobn': [True,False],
    'nobias':[True,False]
}

param_space['SIGNET'] = {
    'h_feats': [16, 32, 64,128],
    'encoder_layers': [1, 2, 3,4,5],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
    'pooling': ['max','add'],
    'graph_level_pool':['global_mean_pool','global_max_pool'],
    'readout':['concat','add','last'],
    'explainer_model':['mlp','gin'],
    'explainer_layers':[1, 2, 3,4,5],
    'explainer_hidden_dim':[8,16,32],
    'explainer_readout':['concat','add','last']
}

param_space['GLocalKD'] = {
    'h_feats': [16, 32, 64,128],
    'num_layers': [1, 2, 3,4,5],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
    'clip': [0.1,0.15,0.20],
    'nobn': [True,False],
    'nobias':[True,False],
    'output_dim':[64,128,256]
}

param_space['OCGTL'] = {
    'h_feats': [16, 32, 64,128],
    'num_layers': [1, 2, 3,4,5],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}

param_space['OCGIN'] = {
    'h_feats': [16, 32, 64,128],
    'num_layers': [1, 2, 3,4,5],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}
param_space['GraphCL_IF'] = {
    'n_trees': [200,250,300],
    'sample_ratio': [0.3,0.4,0.5,0.6],

}
param_space['InforGraph_IF'] = {
    'n_trees': [200,250,300],
    'sample_ratio': [0.3,0.4,0.5,0.6],
}
param_space['GraphCL_OCSVM'] = {
    'n_trees': [200,250,300],
    'sample_ratio': [0.3,0.4,0.5,0.6],
}
param_space['InforGraph_OCSVM'] = {
    'n_trees': [200,250,300],
    'sample_ratio': [0.3,0.4,0.5,0.6],
}
param_space['KernelGLAD'] = {
    'n_trees': [200,250,300],
    'sample_ratio': [0.3,0.4,0.5,0.6],
    'n_neighbors':[20,30,40],
    'n_leaf':[30,35,40],
    'WL_iter':[3,4,5,6,7]
}


