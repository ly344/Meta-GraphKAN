from typing import List

from torch_geometric.data import Data, Batch

from structuralremap_construction.data.bondgraph.to_bondgraph import to_bondgraph
from structuralremap_construction.data.featurization.mol_to_data import mol_to_data
from structuralremap_construction.data.featurization.smiles_to_3d_mol import smiles_to_3d_mol


import copy



def smiles_to_data(smiles: str) -> Data:
    mol = smiles_to_3d_mol(smiles) 
    data = mol_to_data(mol)  
    data = to_bondgraph(data)  
    data.pos = None  
    return data




from typing import List
from copy import deepcopy
from torch_geometric.data import Data, Batch




from typing import List
from copy import deepcopy
from torch_geometric.data import Batch

def collate_with_circle_index(data_batch: Batch) -> Batch:
    """
    Collates a Batch object into a new Batch object with updated attributes.

    Args:
        data_batch: A Batch object that contains multiple Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).
    """

    if not isinstance(data_batch, Batch):
        raise ValueError("data_batch must be a Batch object.")

    data_list = data_batch.to_data_list()
    
    batch1 = deepcopy(data_list)

    for b in batch1:

        b.x = b.x1
        # b.edge_attr = b.edge_attr1
    
    try:
        batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])
        print(f'batch1: ', batch1)
    except Exception as e:
        print(f"Error creating batch1: {e}")
        raise

    
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])
    print(f'batch: ', batch)

    
    batch.batch1 = batch1.batch 
    batch.edge_index1 = batch1.edge_index1
    return batch


def collate_with_circle_index2(data_list: List[Data]) -> Batch:
    """
    Collates a list of Data objects into a Batch object.

    Args:
        data_list: a list of Data objects. Each Data object must contain `circle_index` attribute.
        k_neighbors: number of k consecutive neighbors to be used in the message passing step.

    Returns:
        Batch object containing the collate attributes from data objects, including `circle_index` collated
        to shape (total_num_nodes, max_circle_size).

    """
    
    batch1 = copy.deepcopy(data_list)  # 创建一个深拷贝
    # print(f'before transform batch1: ', batch1)
    for b in batch1:

        b.x = b.x1  # 处理 x 属性
        # print(b)

    batch1 = Batch.from_data_list(batch1, exclude_keys=['circle_index'])  # 创建第一个 Batch 对象
    batch = Batch.from_data_list(data_list, exclude_keys=['circle_index'])  # 创建第二个 Batch 对象
    # print(f'batch1: ', batch1)
    # print(f'batch: ', batch)

    batch.batch1 = batch1.batch  # 复制 batch 属性
    batch.edge_index1 = batch1.edge_index1  # 复制 edge_index1 属性
    # print(f'transform batch1: ', batch1)
    # print(f'transform batch: ', batch)
    return batch