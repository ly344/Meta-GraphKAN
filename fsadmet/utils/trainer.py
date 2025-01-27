#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fsadmet/trainer.py
# Project: /home/richard/projects/fsadmet/utils
# Created Date: Wednesday, June 29th 2022, 1:34:26 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Fri Jun 07 2024
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
import os
import sys
# import nni
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import numpy as np
from structuralremap_construction import *
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch_geometric.data import  Data
from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader as DataLoaderChem
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)
import torch.optim as optim
import torch

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from fsadmet.dataset.dataset import MoleculeDataset
from fsadmet.dataset.dataset_chem import MoleculeDataset as MoleculeDatasetChem
from fsadmet.dataset.utils import my_collate_fn
from fsadmet.model.samples import sample_datasets, sample_test_datasets
from .train_utils import update_params
from sklearn.metrics import f1_score, average_precision_score
from .loss import LossGraphFunc, LossSeqFunc
from .std_logger import Logger
from .FocalLoss import BCEFocalLoss
from concurrent.futures import ThreadPoolExecutor

import nni


#待删除
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from torch.nn.functional import relu, softmax, elu
from ana.tsne_analysis import TSNEModule,PCAModule

from nt_xent import NT_Xent

class Trainer(object):

    def __init__(self, meta_model, cfg, device):

        self.meta_model = meta_model # 保存传入的元学习模型
        self.device = device
        self.cfg = cfg
        # 初始化数据集名称和数据路径
        # meta-learning parameters
        self.dataset_name = cfg.data.dataset
        self.data_path_root = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            cfg.data.path)
        # 初始化任务数量和N-way分类的参数
        self.num_tasks = cfg.tasks[self.dataset_name].num_tasks
        self.num_train_tasks = len(cfg.tasks[self.dataset_name].train_tasks)
        self.num_test_tasks = len(cfg.tasks[self.dataset_name].test_tasks)
        self.n_way = cfg.tasks[self.dataset_name].n_way
        self.m_support = cfg.tasks[self.dataset_name].m_support
        self.k_query = cfg.tasks[self.dataset_name].k_query
        # 初始化训练参数
        # training paramters
        self.batch_size = cfg.train.batch_size
        self.meta_lr = cfg.train.meta_lr
        self.update_lr = cfg.train.update_lr
        self.update_step = cfg.train.update_step
        self.update_step_test = cfg.train.update_step_test
        self.eval_epoch = cfg.train.eval_epoch

        self.saved_model_metric = 0
        self.default_metric = 0
        self.default_f1 = 0
        self.default_pr_auc = 0
        if cfg.model.backbone == "gnn":

            self.optimizer = optim.Adam(
                self.meta_model.base_model.parameters(),
                lr=cfg.train.meta_lr,
                weight_decay=cfg.train.decay)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        # # 默认gama是4
        # self.criterion = BCEFocalLoss(gamma=4, alpha=0.25, reduction='elementwise_mean')

        # optimizer = torch.optim.Adam(model.parameters(), lr=self.update_lr)
        self.epochs = cfg.train.epochs
        self.num_workers = cfg.data.num_workers

    def train_epoch(self, epoch):
        # 采样数据加载器
        # samples dataloaders
        support_loaders = []
        query_loaders = []

        self.meta_model.base_model.train()
        # 遍历训练任务，创建数据加载器
        for task in self.cfg.tasks[self.dataset_name].train_tasks:
            # for task in tasks_list:

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(self.data_path_root,
                                                       self.dataset_name,
                                                       "new", str(task+1)),
                                          dataset=self.dataset_name)

                # 当你设置 collate_fn=None 时，意味着使用默认的合并方式。在这种情况下，DataLoader 会使用默认的行为来合并数据。
                collate_fn = None
                MyDataLoader = DataLoader

            support_dataset, query_dataset = sample_datasets(
                dataset, self.dataset_name, task, self.n_way, self.m_support,
                self.k_query)

            support_loader = MyDataLoader(support_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=True,
                                          num_workers=self.num_workers,
                                          collate_fn=collate_fn,
                                          )

            query_loader = MyDataLoader(query_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        collate_fn=collate_fn,
                                       )
            

            
            support_loaders.append(support_loader)
            query_loaders.append(query_loader)

        for k in range(0, self.update_step):

            old_params = parameters_to_vector(
                self.meta_model.base_model.parameters())

            # use this loss to save all the losses of query set
            losses_q = torch.tensor([0.0]).to(self.device)


            # losses_q = []

            for task in range(self.num_train_tasks):

                losses_s = torch.tensor([0.0]).to(self.device)
                # 训练支持集
                # training support
                for _, batch in enumerate(
                        tqdm(
                            support_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration"
                            .format(epoch, k, task + 1))):

                    if self.cfg.model.backbone == "gnn":
                        lossz = 0
                        batch = batch.to(self.device) 
                        # 新加
                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
                        # 计算主要任务的损失。
                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())
                        # 定义两个对比损失函数，用于无监督学习任务。
                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)
                        # 计算了辅助任务的损失，类似于主要任务的损失计算。
                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())
                        # 分别计算了节点级别的对比损失和图级别的对比损失。
                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)
                        # loss = loss_2
                        # 这行代码计算了总损失，它是主要任务损失、辅助任务损失和节点对比损失的加权和。
                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        losses_s += loss

                    # 日志记录
                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        f"Training/support_task_{task + 1}_loss",
                        losses_s.item(),
                        step=epoch * self.update_step + k
                    )                
                # 更新参数
                _, new_params = update_params(self.meta_model.base_model,
                                                losses_s,
                                                update_lr=self.update_lr)
                
                # 更新基础模型的参数
                vector_to_parameters(new_params,
                                        self.meta_model.base_model.parameters())
                # print('Train epoch: {} \tLoss: {:.6f}'.format(epoch, lossz))


                this_loss_q = torch.tensor([0.0]).to(self.device)

                # 训练查询集
                # training query task set
                for _, batch in enumerate(
                        tqdm(
                            query_loaders[task],
                            desc=
                            "Training | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration"
                            .format(epoch, k, task + 1))):

                    if self.cfg.model.backbone == "gnn":
                        batch = batch.to(self.device) 
                        # 新加
                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
                        # 计算主要任务的损失。
                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())
                        # 定义两个对比损失函数，用于无监督学习任务。
                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)
                        # 计算了辅助任务的损失，类似于主要任务的损失计算。
                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())
                        # 分别计算了节点级别的对比损失和图级别的对比损失。
                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)
                        # loss = loss_2
                        # 这行代码计算了总损失，它是主要任务损失、辅助任务损失和节点对比损失的加权和。
                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        this_loss_q += loss


                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        f"Training/query_task_{task + 1}_loss",
                        this_loss_q.item(),
                        step=epoch * self.update_step + k)

                
                # 更新所有任务的损失
                if task == 0:
                    losses_q = this_loss_q
                else:
                    losses_q = torch.cat((losses_q, this_loss_q), 0)

                # 应用旧参数到模型中
                vector_to_parameters(old_params, self.meta_model.base_model.parameters())

            # 计算所有任务的平均损失
            loss_q = torch.sum(losses_q) / self.num_train_tasks

            # 记录加权查询损失
            if not self.cfg.mode.nni and self.cfg.logger.log:
                mlflow.log_metric(
                    "Training/weighted_query_loss",
                    loss_q.item(),
                    step=epoch * self.update_step + k)

            # 清零梯度
            self.optimizer.zero_grad()
            # 反向传播
            loss_q.backward()
            # 更新权重
            self.optimizer.step()

            return []

    def test(self, epoch):

        accs = []
        rocs = []
        f1_scores = []
        pr_aucs = []


        all_features = []  # 用于存储所有任务的特征
        all_labels = []    # 用于存储所有任务的真实标签
        old_params = parameters_to_vector(
            self.meta_model.base_model.parameters())

        for task in self.cfg.tasks[self.dataset_name].test_tasks:

            if self.cfg.model.backbone == "gnn":
                dataset = MoleculeDataset(os.path.join(self.data_path_root,
                                                       self.dataset_name,
                                                       "new", str(task+1)),
                                          dataset=self.dataset_name)
                collate_fn = None
                MyDataLoader = DataLoader


            support_dataset, query_dataset = sample_test_datasets(
                dataset, self.dataset_name, task,
                self.n_way, self.m_support, self.k_query)

            support_loader = MyDataLoader(support_dataset,
                                          batch_size=self.batch_size,
                                          shuffle=False,
                                          num_workers=self.num_workers,
                                          collate_fn=collate_fn,
                                          )
            query_loader = MyDataLoader(query_dataset,
                                        batch_size=self.batch_size,
                                        shuffle=False,
                                        num_workers=self.num_workers,
                                        collate_fn=collate_fn,
                                        )

            self.meta_model.eval()

            for k in range(0, self.update_step_test):
                lossz = torch.tensor([0.0]).to(self.device)


                for step, batch in enumerate(
                        tqdm(
                            support_loader,
                            desc=
                            "Testing | Epoch: {} | UpdateK: {} | Task: {} | Support Iteration"
                            .format(epoch, k, task))):

                    if self.cfg.model.backbone == "gnn":
                        
                        batch = batch.to(self.device) 
                        # 新加
                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
                        # 计算主要任务的损失。
                        loss_1 = self.criterion(output, data.y.view(-1, 1).float())
                        # 定义两个对比损失函数，用于无监督学习任务。
                        criterion1 = NT_Xent(output.shape[0], 0.1, 1)
                        criterion2=NT_Xent(output.shape[0], 0.1, 1)
                        # 计算了辅助任务的损失，类似于主要任务的损失计算。
                        loss_2 = self.criterion(output1, data.y.view(-1, 1).float())
                        # 分别计算了节点级别的对比损失和图级别的对比损失。
                        cl_loss_node = criterion1(x_g, x_g1)
                        cl_loss_graph=criterion2(output,output1)
                        # loss = loss_2
                        # 这行代码计算了总损失，它是主要任务损失、辅助任务损失和节点对比损失的加权和。
                        loss = loss_1+loss_2+(0.1*cl_loss_node)+(0.1*cl_loss_graph)

                        lossz += loss

                if not self.cfg.mode.nni and self.cfg.logger.log:
                    mlflow.log_metric(
                        "Testing/support_task_{}_loss".format(task),
                        lossz.item(),
                        step=epoch * self.update_step_test + k)
                new_grad, new_params = update_params(
                    self.meta_model.base_model, lossz, update_lr=self.update_lr)

                vector_to_parameters(new_params,
                                     self.meta_model.base_model.parameters())

            y_true = []
            y_scores = []
            y_predict = []


            for _, batch in enumerate(
                    tqdm(
                        query_loader,
                        desc=
                        "Testing | Epoch: {} | UpdateK: {} | Task: {} | Query Iteration"
                        .format(epoch, k, task))):

                if self.cfg.model.backbone == "gnn":
                    batch = batch.to(self.device)
                    with torch.no_grad():
                        # 新加
                        batch.x = batch.x.to(torch.long)
                        batch.edge_index = batch.edge_index.to(torch.long)
                        batch.edge_attr = batch.edge_attr.to(torch.long)
                        batch.x1 = batch.x1.to(torch.long)
                        batch.edge_index1 = batch.edge_index1.to(torch.long)
                        batch.edge_attr1 = batch.edge_attr1.to(torch.long)
                        data = collate_with_circle_index(batch)
                        data.edge_attr = None
                        batch1 = data.batch1.detach()
                        edge = data.edge_index1.detach()
                        xd = data.x1.detach()
                        # self.optimizer.zero_grad()
                        output, x_g, x_g1, output1 = self.meta_model.base_model(data, data.x, data.edge_index, data.batch, xd, edge, batch1)
                        print('output:',output)

                        # 收集特征和标签
                        # features = torch.cat([x_g, x_g1], dim=1).cpu().numpy()  # 合并两个特征表示
                        features = x_g1.cpu().numpy()
                        labels = batch.y.cpu().numpy()
                        all_features.append(features)
                        all_labels.append(labels)




                        
                        y_score = torch.sigmoid(output.squeeze()).cpu()
                        y_scores.append(y_score)
                        y_predict.append(
                            torch.tensor([1 if _ > 0.5 else 0 for _ in y_score],
                                            dtype=torch.long).cpu())

                        y_true.append(batch.y.cpu())



            y_true = torch.cat(y_true, dim=0).numpy()
            y_scores = torch.cat(y_scores, dim=0).numpy()
            y_predict = torch.cat(y_predict, dim=0).numpy()

            
            # 打印 y_scores 以进一步检查
            print(f"y_scores: {y_scores}")
            print(f"y_true: {y_true}")

            unique_classes = set(y_true)
            if len(unique_classes) == 1:
                print("Only one class present in y_true. ROC AUC score is not defined in that case.")
                roc_score = 0.5  # 或者返回一个默认值，如 0.5
            else:
                roc_score = roc_auc_score(y_true, y_scores)
            # roc_score = roc_auc_score(y_true, y_scores)
            acc_score = accuracy_score(y_true, y_predict)

            # 新加评估指标
            f1 = f1_score(y_true, y_predict)
            pr_auc = average_precision_score(y_true, y_scores)

            if not self.cfg.mode.nni and self.cfg.logger.log:
                # mlflow.log_metric("Testing/query_task_{}_acc".format(task),
                #                   acc_score,
                #                   step=epoch)
                mlflow.log_metric("Testing/query_task_{}_auc".format(task),
                                    roc_score,
                                    step=epoch)
            accs.append(acc_score)
            rocs.append(roc_score)
            f1_scores.append(f1)
            pr_aucs.append(pr_auc)
        if not self.cfg.mode.nni and self.cfg.logger.log:
            # mlflow.log_metric("Testing/query_mean_acc",
            #                   np.mean(accs),
            #                   step=epoch)
            mlflow.log_metric("Testing/query_mean_auc",
                              np.mean(rocs),
                              step=epoch)

        vector_to_parameters(old_params,
                            self.meta_model.base_model.parameters())


        # 将所有特征和标签合并为一个数组
        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        
        
        return accs, rocs,f1_scores, pr_aucs, all_features, all_labels



    def run(self):

        tsne_module = TSNEModule(output_dir='/root/codes/MolFeSCue-master-2/tsne_visualizations/')
        pca_module = PCAModule(output_dir='/root/codes/MolFeSCue-master-2/tsne_visualizations/')

        # 用于遍历每个训练周期（epoch）
        for epoch in range(1, self.epochs + 1):

            self.train_epoch(epoch)
                # 在每个 epoch 结束后清空缓存
            # torch.cuda.empty_cache()
            # 用于决定是否在当前epoch进行评估。 self.eval_epoch是类的一个属性，表示每隔多少个epoch进行一次评估。
            if epoch % self.eval_epoch == 0:
                accs, rocs,f1_scores, pr_aucs,all_features, all_labels = self.test(epoch)
                # 计算所有AUC值的平均值，并保留三位小数。
                mean_roc = round(np.mean(rocs), 4)
                mean_f1 = round(np.mean(f1_scores), 4)
                mean_auc = round(np.mean(pr_aucs), 4)

                # 如果当前的平均AUC值高于之前记录的最高值
                if mean_roc > self.default_metric:
                    self.default_metric = mean_roc
                    self.default_f1 = mean_f1
                    self.default_pr_auc = mean_auc

                # Logger.info("downstream task accs: {}".format(
                #     [round(_, 3) for _ in accs]))
                Logger.info("downstream task aucs: {}".format(
                    [round(_, 4) for _ in rocs]))
                Logger.info(
                    "mean downstream task mean auc、f1、pr_auc: {},{},{}".format(mean_roc, mean_f1, mean_auc))
                
                    # 在这里添加保存模型的代码，当当前平均ROC值高于之前保存的模型对应的平均ROC值时保存模型
                if mean_roc > self.saved_model_metric:  # 假设新增一个属性来记录之前保存模型时的平均ROC值
                    torch.save(self.meta_model.base_model.state_dict(), '/root/codes/MolFeSCue-master-2/model_output/model_epoch_{}.pth'.format(epoch))
                    self.saved_model_metric = mean_roc  # 更新保存模型时的平均ROC值记录



                # 查是否启用了NNI。NNI是一个用于神经网络自动化调优的工具。
                if self.cfg.mode.nni:
                    nni.report_intermediate_result({"default_auc": np.mean(rocs)})


                            # 使用 TSNEModule 进行 t-SNE 可视化
                # tsne_module.visualize(all_features, all_labels, epoch=epoch, 
                #                     title=f't-SNE visualization of the graph embeddings (Epoch {epoch})',
                #                     filename=f't-SNE_visualization_epoch_{epoch}.png')
                pca_module.visualize(all_features, all_labels, epoch=epoch, 
                                    title=f't-SNE visualization of the graph embeddings (Epoch {epoch})',
                                    filename=f't-SNE_visualization_epoch_{epoch}.png')
        # 在训练结束后，向NNI报告最终结果，这里报告的是记录的最高平均AUC值。
        nni.report_final_result({
        "default_auc": self.default_metric,
        "f1": self.default_f1,
        "pr_auc": self.default_pr_auc
        })
  
