#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/dataset/test.py
# Project: /home/richard/projects/fsadmet/dataset
# Created Date: Tuesday, June 28th 2022, 3:27:07 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jun 04 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2022 HILAB
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###
from xml.etree.ElementTree import canonicalize
from rdkit import Chem

import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
import pandas as pd

import os
from itertools import repeat, product, chain

from .utils import _load_tox21_dataset, _load_sider_dataset
import sys
import os
# 获取当前脚本所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取上一级目录的路径
parent_dir = os.path.dirname(current_dir)
# 将上一级目录添加到sys.path中
sys.path.append(parent_dir)
# 现在可以导入上级目录中的包了

from structuralremap_construction.data.bondgraph.to_bondgraph import to_bondgraph
from structuralremap_construction.data.featurization.mol_to_data import mol_to_data
from structuralremap_construction.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol

allowable_features = {
    # 描述了可能的原子序数列表，从1到118。这些数字对应于元素周期表中的元素。
    'possible_atomic_num_list':
    list(range(1, 119)),
    # 列出了可能的形式电荷，从-5到+5。形式电荷是指原子在分子中所带的电荷，可以是正的、负的或零。
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    # 描述了可能的手性类型，包括未指定的手性（CHI_UNSPECIFIED）、四面体碳原子的手性（CHI_TETRAHEDRAL_CW 和 CHI_TETRAHEDRAL_CCW，
    # 分别表示顺时针和逆时针）、以及其他类型的手性（CHI_OTHER）。
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    # 列出了可能的杂化类型，包括S（sp杂化）、SP（sp杂化）、SP2、SP3、SP3D、SP3D2和未指定（UNSPECIFIED）。
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S, Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    # 描述了原子可能结合的氢原子数量，从0到8。
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    # 列出了可能的隐式价，从0到6。隐式价是指原子在分子中可能形成的化学键的数量。
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    # 描述了原子可能的度数，即与原子相连的键的数量，从0到10。
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    # 列出了可能的键类型，包括单键（SINGLE）、双键（DOUBLE）、三键（TRIPLE）和芳香键（AROMATIC）。
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    # 描述了双键的立体化学方向，包括无方向（NONE）、向上（ENDUPRIGHT）和向下（ENDDOWNRIGHT）。这些特征用于表示双键的立体化学特性。
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}


def mol_to_graph_data(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # 定义原子特征的数量，这里只有原子类型和手性标签两种特征
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    #  遍历分子中的每个原子
    for atom in mol.GetAtoms():
        #  获取原子的特征，包括原子序数和手性标签，并将其转换为索引
        atom_feature = [
            allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())
        ] + [
            allowable_features['possible_chirality_list'].index(
                atom.GetChiralTag())
        ]
        atom_features_list.append(atom_feature)
    # 将原子特征列表转换为 PyTorch 张量
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds# 定义键特征的数量，这里只有键类型和键方向两种特征
    num_bond_features = 2  # bond type, bond direction
    # 检查分子是否有键
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        # 获取键的特征，包括键类型和键方向，并将其转换为索引
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [
                allowable_features['possible_bonds'].index(bond.GetBondType())
            ] + [
                allowable_features['possible_bond_dirs'].index(
                    bond.GetBondDir())
            ]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            # 由于无向图，需要添加反向边
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
        # 将边的连接列表转换为 PyTorch 张量
        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        # 将边的特征列表转换为 PyTorch 张量
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                    dtype=torch.long)
    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    # 创建图数据对象，包含原子特征、边连接和边特征
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, SMILES = Chem.MolToSmiles(mol, canonical = True ))

    return data


# 定义了一个名为 MoleculeDataset 的类，它继承自 PyTorch Geometric 的 InMemoryDataset 类。这个类用于处理分子数据集，将分子结构转换为图数据格式
class MoleculeDataset(InMemoryDataset):
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
    def __init__(self,
                 root,
                 dataset,
                 
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 empty=False):


        self.root = root
        self.dataset = dataset
        
        self.task_counts = [0, 0]  # 负样本计数, 正样本计数

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        if not empty:
            self.data, self.slices = torch.load(self.processed_paths[0])

    # @property 是一个装饰器，用于将一个方法转变为属性。
    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list # 返回原始数据目录中的文件列表

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt' 

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')  # 禁用下载功能
    
    # 当你第一次使用 MoleculeDataset 类并尝试访问数据时，如果处理过的文件不存在，
    # __getitem__ 或 download 方法会触发 process 方法的调用。如果处理过的文件已经存在，它将直接加载处理过的数据，而不会再次调用 process 方法。
    # def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == "tox21" or self.dataset == "muv":
            smiles_list, rdkit_mol_objs, labels = \
                _load_tox21_dataset(self.raw_paths[0])
            print(np.unique(labels))
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                ## convert aromatic bonds to double bonds
                #Chem.SanitizeMol(rdkit_mol,
                #sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE)
                data = mol_to_data(rdkit_mol)
                # 检查生成的数据是否具有边。如果没有边（即 edge_index 的形状为 (2, 0)），则打印警告并跳过该分子。
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            print(f'data_smile_list : ', data_smiles_list)
            print(f'data_list : ', data_list[111].y)

            
                
        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = \
                _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                #print(i)
                rdkit_mol = rdkit_mol_objs[i]
                # 手动添加分子 ID
                data = mol_to_data(rdkit_mol)
                # 检查生成的数据是否具有边。如果没有边（即 edge_index 的形状为 (2, 0)），则打印警告并跳过该分子。
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                # manually add mol id
                data.id = torch.tensor([i])  # id here is the index of the mol in
                # the dataset
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
            print(f'data_smile_list : ', data_smiles_list)

        else:
            raise ValueError('Invalid dataset name')
        # # 进行数据过滤
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]

        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]


        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        # 例如：/root/codes/MolFeSCue-master-2/fsadmet/data/sider/new/2/processed/
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               'smiles.csv'),
                                  index=False,
                                  header=False)
        # self.collate将所有数据组合在一起,加速存储
        # data是组合之后的数据
        # slices是分割方式，告诉PyG如何将data还原为原先的数据
        data, slices = self.collate(data_list)
        # 使用PyTorch的 save 函数将打包好的数据和切片信息保存到文件中。保存的文件路径由 self.processed_paths[0] 指定
        torch.save((data, slices), self.processed_paths[0])


    def process(self):
        data_smiles_list = []
        data_list = []
        task_counts = [0,0]  # 初始化任务计数器列表

        if self.dataset == "tox21" or self.dataset == "muv":
            smiles_list, rdkit_mol_objs, labels = _load_tox21_dataset(self.raw_paths[0])
            # print(np.unique(labels))
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_data(rdkit_mol)
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                # data.y = torch.tensor(labels[i], dtype=torch.long)  # 假设 labels 是一维数组

                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

                # 更新任务计数器
                label = data.y.item()
                if label == 0:
                    task_counts[0] += 1  # 负样本计数
                else:
                    task_counts[1] += 1  # 正样本计数

        elif self.dataset == 'sider':
            smiles_list, rdkit_mol_objs, labels = _load_sider_dataset(self.raw_paths[0])
            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_data(rdkit_mol)
                if data.edge_index.shape[1] == 0:
                    print(f"WARNING: Skipping molecule {i} because it has no edges.")
                    continue
                data = to_bondgraph(data)
                data.pos = None
                data.id = torch.tensor([i])
                data.y = torch.tensor(labels[i, :])
                data_list.append(data)
                data_smiles_list.append(smiles_list[i])

                # 更新任务计数器
                label = data.y.item()
                if label == 0:
                    task_counts[0] += 1  # 负样本计数
                else:
                    task_counts[1] += 1  # 正样本计数
    

        else:
            raise ValueError('Invalid dataset name')

        # 进行数据过滤和转换
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]

        # 写入数据
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        self.update_accumulated_counts(task_counts)

        print('Task counts:', task_counts)
        print("Total accumulated task counts:", self.task_counts)

    def update_accumulated_counts(self, task_counts=None):
        if task_counts is not None:
            self.task_counts[0] += task_counts[0]
            self.task_counts[1] += task_counts[1]
        print("Updated accumulated task counts:", self.task_counts)

    def get_task_counts(self):
        return self.task_counts

    def display_accumulated_counts(self):
        print("Total accumulated task counts:", self.task_counts)



# 该方法用于从 InMemoryDataset 类的一个实例中获取特定索引的数据。下面是对这段代码的详细解释及
# 这对于在训练过程中按索引访问数据集的单个元素非常有用，尤其是在使用 PyTorch Geometric 进行图神经网络训练时。
    def get(self, idx):
        # 创建一个新的 Data 对象，这通常是 PyTorch Geometric 库中用于存储图数据的对象。
        res_data = Data()
    # 遍历数据集中所有数据项的键（这些键对应于不同的数据属性，如节点特征、边索引等）。
        for key in self.data.keys():
            # 获取与键 key 对应的数据项 item 和对应的切片对象 slices。self.data 存储了数据项，而 self.slices 存储了每个数据项在批处理中的切片信息。
            item, slices = self.data[key], self.slices[key]
            # 检查 item 是否是一个 PyTorch 张量。如果是，执行以下操作：
            if isinstance(item, torch.Tensor):
                # 创建一个与 item 维度相同的切片列表，每个维度都是一个全切片（即选择该维度的所有元素）。
                s = list(repeat(slice(None), item.dim()))
                # 获取 res_data 对象中对应于 key 的维度索引，并将其设置为从 slices[idx] 到 slices[idx + 1] 的切片。这实际上是从批处理中选择特定的数据子集。
                s[res_data.__cat_dim__(key,
                                       item)] = slice(slices[idx],
                                                      slices[idx + 1])
        #    如果 item 不是张量（可能是一个标量值），则直接使用 slices[idx] 来获取单个元素。
            else:
                s = slices[idx].item()
            # 将切片后的数据项 s 赋值给 res_data 对象的对应键。  
            res_data[key] = item[s]
        # print(res_data)
        return res_data


if __name__ == "__main__":
    task_counts_list = []  # 初始化存储正负样本数量的列表
    # root = "/root/codes/MolFeSCue-master-2/fsadmet/data/tox21/new/{}".format(0)
    # data = MoleculeDataset(root, dataset="tox21")
    # print(data[1])
    # for i in range(12):
    #     root = "/root/codes/MolFeSCue-master-2/fsadmet/data/tox21/new/{}".format(i+1)
    #     dataset = MoleculeDataset(root, dataset="tox21")
    #     task_counts = dataset.get_task_counts()
    #     task_counts_list.append(task_counts)

    for i in range(1,18):
        root = "/root/codes/MolFeSCue-master-2/fsadmet/data/muv/new/{}".format(i)
        dataset = MoleculeDataset(root, dataset="muv")
        task_counts = dataset.get_task_counts()
        task_counts_list.append(task_counts)



    # for i in range(1, 28):
    #         root = f"/root/codes/MolFeSCue-master-2/fsadmet/data/sider/new/{i}"
    #         dataset = MoleculeDataset(root, dataset="sider")
    #         task_counts = dataset.get_task_counts()
    #         task_counts_list.append(task_counts)

    print("Task counts for each dataset:", task_counts_list)

    formatted_task_counts = [[int(num) for num in counts] for counts in task_counts_list]
    print(formatted_task_counts)

