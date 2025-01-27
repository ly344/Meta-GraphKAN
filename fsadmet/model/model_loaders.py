#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/molgen/models/model_loaders.py
# Project: /home/richard/projects/fsadmet/model
# Created Date: Wednesday, October 6th 2021, 12:23:13 am
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Tue Jun 04 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2021 Bodkin World Domination Enterprises
#
# MIT License
#
# Copyright (c) 2021 HILAB
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
import os  # 导入os模块，用于处理操作系统相关的功能，如文件路径操作。

from omegaconf import DictConfig  # 从omegaconf库导入DictConfig，用于配置管理。
import torch  # 导入PyTorch库，用于构建和训练神经网络。

from fsadmet.model.graph_model import GNN_graphpred  # 从当前包中导入GNN_graphpred类，用于图神经网络的预测。
from utils.std_logger import Logger  # 从utils模块中导入Logger类，用于日志记录。
from fsadmet.model.meta_model import MetaGraphModel, MetaSeqModel  # 从当前包中导入MetaGraphModel和MetaSeqModel类，用于元学习模型。
from transformers import AutoModel, AutoTokenizer  # 从transformers库中导入AutoModel和AutoTokenizer，用于处理预训练的Transformer模型和对应的分词器。

from fsadmet.model.models.model import GraphADT

# 函数：模型准备。根据配置文件初始化并返回一个PyTorch模型。
# def model_preperation(orig_cwd: str, cfg: DictConfig) -> torch.nn.Module:
    
#     # 如果配置文件中指定的模型骨干是GNN（图神经网络）。
#     if cfg.model.backbone == "gnn":

#         # 初始化GNN_graphpred模型，传入配置文件中的相关参数。
#         base_learner = GNN_graphpred(cfg.model.gnn.num_layer,  # GNN的层数
#                                      cfg.model.gnn.emb_dim,  # 嵌入维度
#                                      1,  # 输出维度，通常为1（回归任务）或分类数目（分类任务）
#                                      JK=cfg.model.gnn.JK,  # Jumping Knowledge类型
#                                      drop_ratio=cfg.model.gnn.dropout_ratio,  # Dropout比率
#                                       graph_pooling=cfg.model.gnn.graph_pooling,  # 图池化方法
#                                      gnn_type=cfg.model.gnn.gnn_type)  # GNN类型

#         # 如果配置文件中指定了预训练模型。
#         if cfg.model.gnn.pretrained:
#             Logger.info("从 {} 加载预训练模型......".format(cfg.model.gnn.pretrained))
#             # 从指定路径加载预训练模型权重。
#             base_learner.from_pretrained(os.path.join(orig_cwd, cfg.model.gnn.pretrained))

#         # 用图模型和自监督权重初始化MetaGraphModel。
#         model = MetaGraphModel(base_learner,  # GNN模型
#                                cfg.meta.selfsupervised_weight,  # 自监督学习的权重
#                                cfg.model.gnn.emb_dim)  # 嵌入维度

#     # 如果配置文件中指定的模型骨干是序列模型（seq）。
#     elif cfg.model.backbone == "seq":

#         # 使用transformers库加载预训练的Transformer模型。
#         base_learner = AutoModel.from_pretrained(os.path.join(orig_cwd, cfg.model.seq.pretrained))
#         # 加载与预训练模型匹配的分词器。
#         tokenizer = AutoTokenizer.from_pretrained(os.path.join(orig_cwd, cfg.model.seq.pretrained))

#         # 用序列模型和分词器初始化MetaSeqModel。
#         model = MetaSeqModel(base_learner, tokenizer)

#     # 记录模型加载成功的日志。
#     Logger.info("模型加载成功！......\n")
#     Logger.info(model)

#     return model  # 返回构建好的模型



def model_preperation(orig_cwd: str, cfg: DictConfig, args) -> torch.nn.Module:
    
    # 如果配置文件中指定的模型骨干是GNN（图神经网络）。
    if cfg.model.backbone == "gnn":

        # 初始化GNN_graphpred模型，传入配置文件中的相关参数。
        base_learner = GraphADT(args,num_features_xd=128,dropout=0.2,aug_ratio=0.4,weights=[0.4, 0.4, 0.2, 0.6, 0.3, 0.1])
        # 用图模型和自监督权重初始化MetaGraphModel。
        model = MetaGraphModel(base_learner,  # GNN模型
                               cfg.meta.selfsupervised_weight,  # 自监督学习的权重
                               cfg.model.gnn.emb_dim)  # 嵌入维度



    # 记录模型加载成功的日志。
    Logger.info("模型加载成功！......\n")
    Logger.info(model)

    return model  # 返回构建好的模型


