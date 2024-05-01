
from ogb.utils.features import (atom_to_feature_vector,bond_to_feature_vector) 

import torch
import numpy as np

import copy
import pathlib
import pandas as pd
from tqdm import tqdm
from torch_geometric.data import Data

from utils.mol_utils import bond_to_feature_vector as bond_to_feature_vector_non_santize
from utils.mol_utils import atom_to_feature_vector as atom_to_feature_vector_non_santize

from rdkit import Chem
from rdkit.Chem import AllChem


def smiles2graph(smiles_string, sanitize=True):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string, sanitize=sanitize)
        # atoms
        atom_features_list = []
        atom_label = []
        # print('smiles_string', smiles_string)
        # print('mol', Chem.MolToSmiles(mol), 'vs smiles_string', smiles_string)
        for atom in mol.GetAtoms():
            if sanitize:
                atom_features_list.append(atom_to_feature_vector(atom))
            else:
                atom_features_list.append(atom_to_feature_vector_non_santize(atom))

            atom_label.append(atom.GetSymbol())

        x = np.array(atom_features_list, dtype = np.int64)
        atom_label = np.array(atom_label, dtype = np.str)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0: # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                # edge_feature = bond_to_feature_vector(bond)
                if sanitize:
                    edge_feature = bond_to_feature_vector(bond)
                else:
                    edge_feature = bond_to_feature_vector_non_santize(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype = np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype = np.int64)

        else:   # mol has no bonds
            edge_index = np.empty((2, 0), dtype = np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

        graph = dict()
        graph['edge_index'] = edge_index
        graph['edge_feat'] = edge_attr
        graph['node_feat'] = x
        graph['num_nodes'] = len(x)
        
        return graph 

    except:
        return None

def labeled2graphs(raw_dir):
    '''
        - raw_dir: the position where property csv stored,  
    '''
    path_suffix = pathlib.Path(raw_dir).suffix
    if path_suffix == '.csv':
        df_full = pd.read_csv(raw_dir, engine='python')
        df_full.set_index('SMILES', inplace=True)
        print(df_full[:5])
    else:
        raise ValueError("Support only csv.")
    graph_list = []
    for smiles_idx in tqdm(df_full.index[:]):
        graph_dict = smiles2graph(smiles_idx)
        props = df_full.loc[smiles_idx]
        for (name,value) in props.iteritems():
            graph_dict[name] = np.array([[value]])
        graph_list.append(graph_dict)
    return graph_list

def read_graph_list(raw_dir, property_name=None, drop_property=False, process_labeled=False):
    print('raw_dir', raw_dir)
    assert process_labeled
    graph_list = labeled2graphs(raw_dir)

    pyg_graph_list = []
    print('Converting graphs into PyG objects...')
    for graph in graph_list:
        g = Data()
        g.__num_nodes__ = graph['num_nodes']
        g.edge_index = torch.from_numpy(graph['edge_index'])
        del graph['num_nodes']
        del graph['edge_index']
        if process_labeled:
            g.y = torch.from_numpy(graph[property_name.split('-')[1]])
            del graph[property_name.split('-')[1]]

        if graph['edge_feat'] is not None:
            g.edge_attr = torch.from_numpy(graph['edge_feat'])
            del graph['edge_feat']

        if graph['node_feat'] is not None:
            g.x = torch.from_numpy(graph['node_feat'])
            del graph['node_feat']

        addition_prop = copy.deepcopy(graph)
        for key in addition_prop.keys():
            g[key] = torch.tensor(graph[key])
            del graph[key]

        pyg_graph_list.append(g)

    return pyg_graph_list