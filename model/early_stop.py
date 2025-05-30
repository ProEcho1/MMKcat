#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024-02-29 15:01
# @Author  : Xin Sun
# @ID      : 22371220
# @File    : early_stop.py
# @Software: PyCharm

import torch

import os
import numpy as np


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(),
                   os.path.join(path, str(self.dataset) + '_best_checkpoint.pth'))
        self.val_loss_min = val_loss
