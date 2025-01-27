#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/model/gnn.py
# Project: /home/richard/projects/fsadmet/model
# Created Date: Tuesday, June 28th 2022, 6:22:32 pm
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

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros

from .layer import GINConv, GCNConv, GraphSAGEConv, GATConv
from torch_geometric.nn import SAGPooling, TopKPooling

class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 JK="last",
                 drop_ratio=0.5,
                 num_atom_type=120,
                 num_chirality_tag=3,
                 gnn_type="gin"):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        if self.JK == "concat":
            self.jk_proj = torch.nn.Linear((self.num_layer + 1) * emb_dim,
                                           emb_dim)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        self.topk_pools = torch.nn.ModuleList()  # 添加 TopKPooling 层
        for layer in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim))
            elif gnn_type == "gat":
                self.gnns.append(GATConv(emb_dim))
            elif gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(emb_dim))
            # self.topk_pools.append(TopKPooling(emb_dim, ratio=0.8))  # 添加 TopKPooling 层
        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
         # 初始化节点特征列表
        h_list = [x]
        for layer in range(self.num_layer):
            # 通过 GINConv 层更新节点特征
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            # 应用批量归一化
            h = self.batch_norms[layer](h)
            #h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h),
                              self.drop_ratio,
                              training=self.training)
            
                
            # # 应用 TopKPooling
            # if batch is None:
            #     batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
            
            # # 解包结果
            # result = self.topk_pools[layer](h, edge_index, batch=batch)
            
            # # 确保返回值的数量正确
            # if len(result) != 6:
            #     raise ValueError(f"Unexpected number of return values from TopKPooling: {len(result)}")
            
            # h, edge_index, edge_attr, batch, perm, score = result
            
            # 将更新后的节点特征追加到列表中
            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "concat":
            node_rep = torch.cat(h_list, dim=1)
            # node_rep = 0.6*h_list[0] + 0.3*h_list[1] + 0.1*h_list[2]
            node_rep = self.jk_proj(node_rep)
        elif self.JK == "last":
            node_rep = h_list[-1]
        # 在不同的层之间聚合节点表示，这里采用的是“最大值”（max）聚合策略
        # h_list 是一个列表，其中包含每一层的节点表示张量。
        # 使用 torch.unsqueeze(h, 0) 将每个张量 h 沿着第一个维度（即 dim=0）增加一个新的维度。
        # 结果是一个形状为 [num_layers, num_nodes, emb_dim] 的张量列表。
        elif self.JK == "max":
            h_list = [torch.unsqueeze(h, 0) for h in h_list]
            node_rep = torch.max(torch.cat(h_list, dim=0),
                                 keepdim=False,
                                 dim=0)[0]
        elif self.JK == "sum":
            h_list = [torch.unsqueeze(h, 0) for h in h_list]
            node_rep = torch.sum(torch.cat(h_list, dim=0),
                                 keepdim=False,
                                 dim=0)
        return node_rep


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        
    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self,
                 num_layer,
                 emb_dim,
                 num_tasks,
                 JK="last",
                 drop_ratio=0,
                 graph_pooling="mean",
                 gnn_type="gin",
                 pooling_ratio = 0.8 ):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.num_tasks = num_tasks
        self.num_workers = 2
        self.pooling_ratio = pooling_ratio

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type=gnn_type)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(
                    gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim,
                                    set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        self.graph_pred_linear = torch.nn.Linear(self.mult * self.emb_dim,
                                                 self.num_tasks)

    def from_pretrained(self, model_file):
        # self.gnn = GNN(self.num_layer, self.emb_dim, JK = self.JK, drop_ratio = self.drop_ratio)
        # 加载预训练模型的权重到模型的gnn属性中。
        # torch.load函数用于加载保存的模型权重，map_location='cpu'指定所有张量都映射到CPU上。
        # self.gnn.load_state_dict(torch.load(model_file, map_location='cpu'))

            # 加载预训练模型的权重到模型的gnn属性中。
        pretrained_state_dict = torch.load(model_file, map_location='cpu')

        # 过滤掉预训练模型中与当前模型不匹配的部分
        model_state_dict = self.gnn.state_dict()
        pretrained_filtered = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

        # 检查是否有缺失的键
        missing_keys = set(model_state_dict.keys()) - set(pretrained_filtered.keys())
        if missing_keys:
            print(f'Warning: Missing keys in state_dict: {missing_keys}')

        # 检查是否有意外的键
        unexpected_keys = set(pretrained_state_dict.keys()) - set(model_state_dict.keys())
        if unexpected_keys:
            print(f'Warning: Unexpected keys in state_dict: {unexpected_keys}')

        # 加载过滤后的预训练权重
        self.gnn.load_state_dict(pretrained_filtered, strict=False)

    # def forward(self, *argv):
    #     """
    #     模型的前向传播函数。
    
    #     :param self: 模型实例。
    #     :param argv: 传入的参数，可以是图数据的各个部分或一个包含所有数据的Data对象。
    #     :return: 模型的预测结果，图表示和节点表示。
    #     """

    #     if len(argv) >= 4:
    #         for i, arg in enumerate(argv):
    #             print(f"Argument {i}: {arg}")
    #         # data, data.x, data.edge_index, data.edge_attr,batch.batch, xd, edge, batch1,edge_attr1
    #         # 如果传入了四个参数，分别赋值给x, edge_index, edge_attr, batch。
    #         x, edge_index, edge_attr, batch, x1, edge_index1, batch1,edge_attr1 = argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]
    #     elif len(argv) == 1:
    #         # 如果只传入了一个参数，假设它是一个Data对象，从中提取图数据的各个部分。
    #         data = argv[0]
    #         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
    #     else:
    #         # 如果参数数量不匹配，抛出异常。
    #         raise ValueError("unmatched number of arguments.")
    #     # 调用gnn属性（即图神经网络模型）进行前向传播，得到节点表示。
    #     node_rep = self.gnn(x, edge_index, edge_attr)
    #     # 调用pool方法对节点表示进行图级别的池化，得到图表示。
    #     graph_rep = self.pool(node_rep, batch)
    #      # 调用graph_pred_linear属性（即图预测线性层）对图表示进行处理，得到最终的预测结果。
    #     pred = self.graph_pred_linear(graph_rep)

    #     return pred, graph_rep, node_rep



    def forward(self, *argv):
        """
        模型的前向传播函数。
    
        :param self: 模型实例。
        :param argv: 传入的参数，可以是图数据的各个部分或一个包含所有数据的Data对象。
        :return: 模型的预测结果，图表示和节点表示。
        """

        if len(argv) >= 4:
            # for i, arg in enumerate(argv):
            #     print(f"Argument {i}: {arg}")
            # data, data.x, data.edge_index, data.edge_attr,batch.batch, xd, edge, batch1,edge_attr1
            # 如果传入了四个参数，分别赋值给x, edge_index, edge_attr, batch。
            # x, edge_index, edge_attr, batch, x1, edge_index1, batch1,edge_attr1 = argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]
            
            # 解包参数
            data, x, edge_index, edge_attr, batch, x1, edge_index1, batch1, edge_attr1 = argv
        elif len(argv) == 1:
            # 如果只传入了一个参数，假设它是一个Data对象，从中提取图数据的各个部分。
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            # 如果参数数量不匹配，抛出异常。
            raise ValueError("unmatched number of arguments.")
        # 调用gnn属性（即图神经网络模型）进行前向传播，得到节点表示。
        node_rep = self.gnn(x, edge_index, edge_attr)
        node_rep1 = self.gnn(x1, edge_index1, edge_attr1)

        # 调用pool方法对节点表示进行图级别的池化，得到图表示。
        graph_rep = self.pool(node_rep, batch)
        graph_rep1 = self.pool(node_rep1, batch1)
         # 调用graph_pred_linear属性（即图预测线性层）对图表示进行处理，得到最终的预测结果。
        pred = self.graph_pred_linear(graph_rep)
        pred1 = self.graph_pred_linear(graph_rep1)

        # print(f'pred:', pred)
        # print(f'graph_pred:', graph_rep)
        # print(f'node_rep:', node_rep)

        # print('------------------------------------')
        # print(f'pred:', pred1)
        # print(f'graph_pred:', graph_rep1)
        # print(f'node_rep:', node_rep1)

        return pred, graph_rep, node_rep, pred1, graph_rep1, node_rep1

   


if __name__ == "__main__":
    pass
