import os
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from itertools import repeat
from tqdm import tqdm

list_atoms = ['Rb', 'Mg', 'Kr', 'B', 'P', 'H', 'K', 'Xe', 'Zn', 'Se', 'Ba', 'Cs',
              'Li', 'Na', 'He', 'Al', 'Sr', 'Ga', 'Ra', 'Ca', 'O', 'Be', 'As', 'I',
                'Ag', 'Bi', 'N', 'F', 'C', 'Te', 'Br', 'S', 'Si', 'Cl', 'others', 'masked']
def onehot_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [1 if x == s else 0 for s in allowable_set]


def onehot_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-2]
    return [1 if x == s else 0 for s in allowable_set]

def atom_attr(mol):
    feat = []
    index = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() in list_atoms:
            index.append(list_atoms.index(atom.GetSymbol()))
        else:
            index.append(35)
        results = onehot_encoding_unk(
            atom.GetSymbol(),
            ['Rb', 'Mg', 'Kr', 'B', 'P', 'H', 'K', 'Xe', 'Zn', 'Se', 'Ba', 'Cs',
              'Li', 'Na', 'He', 'Al', 'Sr', 'Ga', 'Ra', 'Ca', 'O', 'Be', 'As', 'I',
                'Ag', 'Bi', 'N', 'F', 'C', 'Te', 'Br', 'S', 'Si', 'Cl', 'others', 'masked']) + onehot_encoding(atom.GetDegree(),
                                  [0, 1, 2, 3, 4, 5, 6, 7, 8]) + \
                  onehot_encoding_unk(atom.GetHybridization(), [
                      Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                      Chem.rdchem.HybridizationType.SP3D2, 'others', 'masked'
                  ]) + onehot_encoding_unk(atom.GetTotalNumHs(),
                                                    [0, 1, 2, 3, 4, 'masked'])
        feat.append(results)

    return np.array(feat), index
 
def bond_attr(mol, use_chirality=True):
    feat = []
    index = []
    n = mol.GetNumAtoms()
    for i in range(n):
        for j in range(n):
            if i != j:
                bond = mol.GetBondBetweenAtoms(i, j)
                if bond is not None:
                    bt = bond.GetBondType()
                    bond_feats = [
                        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
                        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
                        bond.GetIsConjugated(),
                        bond.IsInRing()
                    ]
                    if use_chirality:
                        bond_feats = bond_feats + onehot_encoding_unk(
                            str(bond.GetStereo()),
                            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
                    feat.append(bond_feats)
                    index.append([i, j])

    return np.array(index), np.array(feat)



def mol_to_graph_data_obj_simple(mol):
    if mol is None: return None
    node_attr, index = atom_attr(mol)
    edge_index, edge_attr = bond_attr(mol)
    data = Data(
        x=torch.FloatTensor(node_attr),
        edge_index=torch.LongTensor(edge_index).t(),
        edge_attr=torch.FloatTensor(edge_attr),
        labels_atoms = torch.LongTensor(index),
    )
    return data




class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.root = root
        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
       
        # self.data, self.slices = torch.load(self.processed_paths[0])
        self.data, self.slices = self.process()

    def get(self, idx):
        data = Data()
        for key in ['x', 'edge_index', 'edge_attr', 'id', 'labels_atoms']:
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def process(self):
        data_smiles_list = []
        data_list = []
        smiles_list, rdkit_mol_objs = load_hdac(self.raw_paths[0])
        for i in tqdm(range(len(smiles_list))):
            rdkit_mol = rdkit_mol_objs[i]

            data = mol_to_graph_data_obj_simple(rdkit_mol)
            
            # manually add mol id
            data['id'] = torch.tensor([i])  # id here is the index of the mol in
            
            data_list.append(data)
            data_smiles_list.append(smiles_list[i])


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        return data, slices


def load_data_extracted(input_path):
    input_df = pd.read_csv(input_path)
    smiles_list = input_df['SMILES'].tolist()
    smiles_list = smiles_list
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
    data = []
    for rdkit_mol in rdkit_mol_objs_list:
        data.append(mol_to_graph_data_obj_simple(rdkit_mol))
    return data


def load_hdac(input_path):
    input_df = pd.read_csv(input_path)
    smiles_list = input_df['Canonical Smiles'].tolist()
    smiles_list = smiles_list
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
   
    assert len(smiles_list) == len(rdkit_mol_objs_list)


    return smiles_list, rdkit_mol_objs_list
