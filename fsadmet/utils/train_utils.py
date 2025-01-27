#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/FSADMET/utils/train_utils.py
# Project: /home/richard/projects/fsadmet/utils
# Created Date: Wednesday, June 29th 2022, 12:12:49 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Sat Dec 02 2023
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

import random
import numpy as np
import torch

# from torch.nn.utils.convert_parameters import (vector_to_parameters,
#                                                parameters_to_vector)
# 这个函数用于更新模型参数。它接收三个参数：base_model（基础模型），loss（损失函数的输出），和 update_lr（更新学习率）。
# def update_params(base_model, loss, update_lr):
#     #  用于计算损失函数关于模型参数的梯度。
#     grads = torch.autograd.grad(loss, base_model.parameters(), allow_unused=True)

#     # Replace None gradients with zeros
#     grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, base_model.parameters())]
#     # 将梯度和参数转换为向量形式。
#     return parameters_to_vector(grads), parameters_to_vector(
#         base_model.parameters(
#         )) - parameters_to_vector(grads) * update_lr

# 这个函数用于构建负样本边，通常用于图神经网络中的对比学习或其他需要负样本的任务。
def build_negative_edges(batch):
    # 提取图中所有边的起始节点和结束节点
    font_list = batch.edge_index[0, ::2].tolist()
    back_list = batch.edge_index[1, ::2].tolist()
    # 创建一个字典，用于存储每个节点的所有邻居节点
    all_edge = {}
    for count, front_e in enumerate(font_list):
         # 如果节点不在字典中，添加它并设置其邻居列表
        if front_e not in all_edge:
            all_edge[front_e] = [back_list[count]]
        else:
             # 如果节点已在字典中，添加新的邻居
            all_edge[front_e].append(back_list[count])

    negative_edges = []
    # 遍历图中的所有节点
    for num in range(batch.x.size()[0]):
        # 如果节点在字典中，即它有邻居
        if num in all_edge:
            # 遍历该节点之后的所有节点，寻找不在其邻居列表中的节点
            for num_back in range(num, batch.x.size()[0]):
                if num_back not in all_edge[num] and num != num_back:
                    negative_edges.append((num, num_back))# 添加负样本边
        else:# 如果节点不在字典中，即它没有邻居
            for num_back in range(num, batch.x.size()[0]):
                if num != num_back:
                    # 添加负样本边
                    negative_edges.append((num, num_back))

    negative_edge_index = torch.tensor(np.array(
        random.sample(negative_edges, len(font_list))).T,
                                       dtype=torch.long)
    # 返回负样本边的索引
    return negative_edge_index

# 这个函数用于更新模型参数。它接收三个参数：base_model（基础模型），loss（损失函数的输出），和 update_lr（更新学习率）。
def update_params(base_model, loss, update_lr):
    # 用于计算损失函数关于模型参数的梯度。
    grads = torch.autograd.grad(loss, base_model.parameters(), allow_unused=True)

    # Replace None gradients with zeros
    grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, base_model.parameters())]

    # 将梯度和参数转换为向量形式。
    grads_vector = parameters_to_vector(grads)
    params_vector = parameters_to_vector(base_model.parameters())

    # 更新参数向量
    updated_params_vector = params_vector - grads_vector * update_lr

    return grads_vector, updated_params_vector

def parameters_to_vector(parameters):
    """Convert parameters to a single vector."""
    # Check that parameters is an iterable
    if not isinstance(parameters, list):
        parameters = list(parameters)
    
    # Use reshape instead of view
    vec = []
    for param in parameters:
        vec.append(param.reshape(-1))
    return torch.cat(vec)

def vector_to_parameters(vector, parameters):
    """Convert a vector back to the parameters."""
    # Check that parameters is an iterable
    if not isinstance(parameters, list):
        parameters = list(parameters)
    
    # Use reshape instead of view
    pointer = 0
    for param in parameters:
        num_param = param.numel()
        param.copy_(vector[pointer:pointer + num_param].reshape(param.shape))
        pointer += num_param




# 这个函数用于构建负样本边，通常用于图神经网络中的对比学习或其他需要负样本的任务。
def build_negative_edges1(batch):
    # 提取图中所有边的起始节点和结束节点
    font_list = batch.edge_index1[0, ::2].tolist()
    back_list = batch.edge_index1[1, ::2].tolist()
    # 创建一个字典，用于存储每个节点的所有邻居节点
    all_edge = {}
    for count, front_e in enumerate(font_list):
         # 如果节点不在字典中，添加它并设置其邻居列表
        if front_e not in all_edge:
            all_edge[front_e] = [back_list[count]]
        else:
             # 如果节点已在字典中，添加新的邻居
            all_edge[front_e].append(back_list[count])

    negative_edges = []
    # 遍历图中的所有节点
    for num in range(batch.x1.size()[0]):
        # 如果节点在字典中，即它有邻居
        if num in all_edge:
            # 遍历该节点之后的所有节点，寻找不在其邻居列表中的节点
            for num_back in range(num, batch.x1.size()[0]):
                if num_back not in all_edge[num] and num != num_back:
                    negative_edges.append((num, num_back))# 添加负样本边
        else:# 如果节点不在字典中，即它没有邻居
            for num_back in range(num, batch.x1.size()[0]):
                if num != num_back:
                    # 添加负样本边
                    negative_edges.append((num, num_back))

    negative_edge_index = torch.tensor(np.array(
        random.sample(negative_edges, len(font_list))).T,
                                       dtype=torch.long)
    # 返回负样本边的索引
    return negative_edge_index


