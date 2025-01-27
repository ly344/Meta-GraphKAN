#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/train.py
# Project: /home/richard/projects/fsadmet
# Created Date: Tuesday, June 28th 2022, 6:47:18 pm
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
import os
import sys
import argparse
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
import mlflow
import nni
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from model.model_loaders import model_preperation
from utils.nni_utils import update_cfg
from utils.helpers import fix_random_seed
from utils.trainer import Trainer
from utils.distribution import setup_multinodes


@hydra.main(config_path="conf", config_name="conf")
def main(cfg: DictConfig):

    # Training settings

    orig_cwd = hydra.utils.get_original_cwd()
    global_rank = 0
    local_rank = 0
    world_size = 0

    if not cfg.mode.nni and cfg.logger.log:
        mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
        mlflow.set_experiment(os.environ["MLFLOW_EXPERIMENT_NAME"])

    if cfg.mode.nni:
        # use nni params
        cfg = update_cfg(cfg)

    fix_random_seed(cfg.train.random_seed, cuda_deterministic=True)

    if cfg.logger.log:

        # log hyper-parameters
        for p, v in cfg.data.items():
            mlflow.log_param(p, v)

        for p, v in cfg.model.items():
            mlflow.log_param(p, v)

        for p, v in cfg.meta.items():
            mlflow.log_param(p, v)

        for p, v in cfg.train.items():
            mlflow.log_param(p, v)

        for p, v in cfg.tasks[cfg.tasks.name].items():
            mlflow.log_param(p, v)

    # device = torch.device("cuda", local_rank)
    # # meta_model = model_preperation(orig_cwd, cfg).to(device)

    # meta_model = model_preperation(orig_cwd, cfg, args = args).to(device)
    

    # device = torch.device("cuda", local_rank)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # # meta_model = model_preperation(orig_cwd, cfg).to(device)
    # meta_model = model_preperation(orig_cwd, cfg, args=args)
    # if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
    #     print(f'-----------------------------------------------')
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     print(f'-----------------------------------------------')

    #     meta_model = nn.DataParallel(meta_model)  # 将模型对象转变为多GPU并行运算的模型

    # meta_model.to(device)  # 把并行的模型移动到GPU上


    # trainer = Trainer(meta_model, cfg, device=device)

    # trainer.run()

    device = torch.device("cuda", local_rank)
    meta_model = model_preperation(orig_cwd, cfg, args=args).to(device)

    trainer = Trainer(meta_model, cfg, device=device)
    trainer.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--seed', type=int, default=42, help='random seed')
    # parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    # parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')

    parser.add_argument('--nhid', type=int, default=128, help='hidden size')

    parser.add_argument('--sample_neighbor', type=bool, default=False, help='whether sample neighbors')
    parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
    parser.add_argument('--structure_learning', type=bool, default=False, help='whether perform structure learning')
    parser.add_argument('--hop_connection', type=bool, default=True, help='whether directly connect node within h-hops')
    parser.add_argument('--hop', type=int, default=3, help='h-hops')
    parser.add_argument('--pooling_ratio', type=float, default=0.8, help='pooling ratio')
    parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--lamb', type=float, default=2.0, help='trade-off parameter')
    # parser.add_argument('--dataset', type=str, default='rabbit', help='DD/PROTEINS/NCI1/NCI109/Mutagenicity/ENZYMES')
    parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
    # parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    # parser.add_argument('--experiment_root', type=str, default='/root/codes/GraphADT/experiment_result/rabbit/noramal/kfold_splits/fold_', help='patience for early stopping')
    # 新加kan
    # parser.add_argument('--layers_hidden', type=int, default=128, help='number of hidden units')
    parser.add_argument('--grid_size', type=int, default=10, help='grid size for KANLinear')
    parser.add_argument('--spline_order', type=int, default=3, help='spline order for KANLinear')
    parser.add_argument('--scale_noise', type=float, default=0.1, help='scale noise for KANLinear')
    parser.add_argument('--scale_base', type=float, default=1.0, help='scale base for KANLinear')
    parser.add_argument('--scale_spline', type=float, default=1.0, help='scale spline for KANLinear')
    parser.add_argument('--base_activation', type=str, default='RELU', help='base activation function for KANLinear')
    parser.add_argument('--grid_eps', type=float, default=0.02, help='grid epsilon for KANLinear')
    parser.add_argument('--grid_range', type=list, default=[-1, 1], help='grid range for KANLinear')
    args = parser.parse_args()
    main()
