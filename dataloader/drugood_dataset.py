import numpy as np
import os.path as osp
import torch
import torch.utils
import torch.utils.data
from torch_geometric.data import InMemoryDataset, Data, DataLoader
import mmcv
import rdkit
from rdkit import Chem

import os
import json
import pickle
from tqdm import tqdm
from .smile2graph import smile2graph4drugood


class SmileToGraph(object):

    """Transform smile input to graph format"""

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = self.smile2graph(results[key])
        return results

    def get_atom_features(self, atom):
        # The usage of features is along with the Attentive FP.
        feature = np.zeros(39)
        # Symbol
        symbol = atom.GetSymbol()
        symbol_list = ['B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br', 'Te', 'I', 'At']
        if symbol in symbol_list:
            loc = symbol_list.index(symbol)
            feature[loc] = 1
        else:
            feature[15] = 1

        # Degree
        degree = atom.GetDegree()
        if degree > 5:
            print("atom degree larger than 5. Please check before featurizing.")
            raise RuntimeError

        feature[16 + degree] = 1

        # Formal Charge
        charge = atom.GetFormalCharge()
        feature[22] = charge

        # radical electrons
        radelc = atom.GetNumRadicalElectrons()
        feature[23] = radelc

        # Hybridization
        hyb = atom.GetHybridization()
        hybridization_list = [rdkit.Chem.rdchem.HybridizationType.SP,
                              rdkit.Chem.rdchem.HybridizationType.SP2,
                              rdkit.Chem.rdchem.HybridizationType.SP3,
                              rdkit.Chem.rdchem.HybridizationType.SP3D,
                              rdkit.Chem.rdchem.HybridizationType.SP3D2]
        if hyb in hybridization_list:
            loc = hybridization_list.index(hyb)
            feature[loc + 24] = 1
        else:
            feature[29] = 1

        # aromaticity
        if atom.GetIsAromatic():
            feature[30] = 1

        # hydrogens
        hs = atom.GetNumImplicitHs()
        feature[31 + hs] = 1

        # chirality, chirality type
        if atom.HasProp('_ChiralityPossible'):
            feature[36] = 1

            try:
                chi = atom.GetProp('_CIPCode')
                chi_list = ['R', 'S']
                loc = chi_list.index(chi)
                feature[37 + loc] = 1
            except KeyError:
                feature[37] = 0
                feature[38] = 0

        return feature

    def get_bond_features(self, bond):
        feature = np.zeros(10)

        # bond type
        type = bond.GetBondType()
        bond_type_list = [rdkit.Chem.rdchem.BondType.SINGLE,
                          rdkit.Chem.rdchem.BondType.DOUBLE,
                          rdkit.Chem.rdchem.BondType.TRIPLE,
                          rdkit.Chem.rdchem.BondType.AROMATIC]
        if type in bond_type_list:
            loc = bond_type_list.index(type)
            feature[0 + loc] = 1
        else:
            print("Wrong type of bond. Please check before feturization.")
            raise RuntimeError

        # conjugation
        conj = bond.GetIsConjugated()
        feature[4] = conj

        # ring
        ring = bond.IsInRing()
        feature[5] = ring

        # stereo
        stereo = bond.GetStereo()
        stereo_list = [rdkit.Chem.rdchem.BondStereo.STEREONONE,
                       rdkit.Chem.rdchem.BondStereo.STEREOANY,
                       rdkit.Chem.rdchem.BondStereo.STEREOZ,
                       rdkit.Chem.rdchem.BondStereo.STEREOE]
        if stereo in stereo_list:
            loc = stereo_list.index(stereo)
            feature[6 + loc] = 1
        else:
            print("Wrong stereo type of bond. Please check before featurization.")
            raise RuntimeError

        return feature

    def smile2graph(self, smile):
        mol = Chem.MolFromSmiles(smile)
        if (mol is None):
            return None
        src = []
        dst = []
        atom_feature = []
        bond_feature = []

        try:
            for atom in mol.GetAtoms():
                one_atom_feature = self.get_atom_features(atom)
                atom_feature.append(one_atom_feature)

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                one_bond_feature = self.get_bond_features(bond)
                src.append(i)
                dst.append(j)
                bond_feature.append(one_bond_feature)
                src.append(j)
                dst.append(i)
                bond_feature.append(one_bond_feature)

            src = torch.tensor(src).long()
            dst = torch.tensor(dst).long()
            edge_index = torch.vstack([src,dst])
            atom_feature = np.array(atom_feature)
            bond_feature = np.array(bond_feature)
            atom_feature = torch.tensor(atom_feature).float()
            bond_feature = torch.tensor(bond_feature).float()
            graph_cur_smile = Data(x=atom_feature, edge_index=edge_index, edge_attr=bond_feature)
            return graph_cur_smile

        except RuntimeError:
            return None

class DrugOOD(InMemoryDataset):
    splits = ['iid', 'ood', 'mixed']

    def __init__(self, root, mode='iid', transform=None, pre_transform=None, pre_filter=None):
        assert mode in self.splits
        self.mode = mode
        self.smile2graph = SmileToGraph(['smiles'])

        super(DrugOOD, self).__init__(root, transform, pre_transform, pre_filter)

        idx = self.processed_file_names.index('drugood_{}.pt'.format(mode))
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        return ['lbap_general_ec50_assay.json']

    @property
    def processed_file_names(self):
        return ['drugood_iid.pt', 'drugood_ood.pt', 'drugood_mixed.pt']

    def download(self):
        if not osp.exists(osp.join(self.raw_dir, 'lbap_general_ec50_assay.json')):
            print("raw data of `DrugOOD` doesn't exist, please redownload from our github.")
            raise FileNotFoundError

    def process(self):
        if self.mode == 'iid':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['train']
            num = 2000
        elif self.mode == 'ood':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_test']
            num = 500
        elif self.mode == 'mixed':
            raw_dataset = mmcv.load(osp.join(self.raw_dir, self.raw_file_names[0]))['split']['ood_val']
            num = 1000
        data_list = []
        for idx, case in enumerate(raw_dataset[:num]):
            case = self.smile2graph(case)
            if self.mode  == 'ood':
                case['smiles'].y = 1
            else:
                case['smiles'].y = 0
            case['smiles'].idx = idx
            data = case['smiles']
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        idx = self.processed_file_names.index('drugood_{}.pt'.format(self.mode))
        torch.save(self.collate(data_list), self.processed_paths[idx])


class DrugOODDataset(InMemoryDataset):
    def __init__(self, name, version='chembl30', type='lbap', root='data', drugood_root='drugood-data',
                 transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.root = root
        # self.dir_name = '_'.join(name.split('-'))
        self.drugood_root = drugood_root
        self.version = version
        self.type = type
        super(DrugOODDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.data_cfg = pickle.load(open(self.processed_paths[1], 'rb'))
        self.data_statistics = pickle.load(open(self.processed_paths[2], 'rb'))
        self.train_index, self.valid_index, self.test_index = pickle.load(open(self.processed_paths[3], 'rb'))
        self.num_tasks = 1

    @property
    def raw_dir(self):
        # return osp.join(self.ogb_root, self.dir_name, 'mapping')
        # return self.drugood_root
        return self.drugood_root + '-' + self.version

    @property
    def raw_file_names(self):
        # return 'lbap_core_' + self.name + '.json'
        return f'{self.type}_core_{self.name}.json'
        # return 'mol.csv.gz'
        # return f'{self.names[self.name][2]}.csv'

    @property
    def processed_dir(self):
        # return osp.join(self.root, self.name, f'{self.decomp}-processed')
        # return osp.join(self.root, self.dir_name, f'{self.decomp}-processed')
        # return osp.join(self.root, f'{self.name}-{self.version}')
        return osp.join(self.root, f'{self.type}-{self.name}-{self.version}')

    @property
    def processed_file_names(self):
        return 'data.pt', 'cfg.pt', 'statistics.pt', 'split.pt'

    def __subprocess(self, datalist):
        processed_data = []
        for datapoint in tqdm(datalist):
            # ['smiles', 'reg_label', 'assay_id', 'cls_label', 'domain_id']
            smiles = datapoint['smiles']
            x, edge_index, edge_attr = smile2graph4drugood(smiles)
            y = torch.tensor([datapoint['cls_label']]).unsqueeze(0)
            if self.type == 'lbap':
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles,
                            reg_label=datapoint['reg_label'], assay_id=datapoint['assay_id'],
                            domain_id=datapoint['domain_id'])
            else:
                protein = datapoint['protein']
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles, protein=protein,
                            reg_label=datapoint['reg_label'], assay_id=datapoint['assay_id'],
                            domain_id=datapoint['domain_id'])

            data.batch_num_nodes = data.num_nodes
            # if self.pre_filter is not None and not self.pre_filter(data):
            #     continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)
            processed_data.append(data)
        return processed_data, len(processed_data)

    def process(self):
        # data_list = []
        json_data = json.load(open(self.raw_paths[0], 'r', encoding='utf-8'))
        data_cfg, data_statistics = json_data['cfg'], json_data['statistics']
        train_data = json_data['split']['train']
        valid_data = json_data['split']['ood_val']
        test_data = json_data['split']['ood_test']
        train_data_list, train_num = self.__subprocess(train_data)
        valid_data_list, valid_num = self.__subprocess(valid_data)
        test_data_list, test_num = self.__subprocess(test_data)
        data_list = train_data_list + valid_data_list + test_data_list
        train_index = list(range(train_num))
        valid_index = list(range(train_num, train_num + valid_num))
        test_index = list(range(train_num + valid_num, train_num + valid_num + test_num))
        torch.save(self.collate(data_list), self.processed_paths[0])
        pickle.dump(data_cfg, open(self.processed_paths[1], 'wb'))
        pickle.dump(data_statistics, open(self.processed_paths[2], 'wb'))
        pickle.dump([train_index, valid_index, test_index], open(self.processed_paths[3], 'wb'))


    def __repr__(self):
        return '{}({})'.format(self.name, len(self))











