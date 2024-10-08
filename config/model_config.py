# -*- coding:utf-8 -*-
"""
Created on Thu. Sep. 26 10:07:11 2024
@author: JUN-SU PARK
"""


class ModelTrainerConfig:
    def __init__(
            self,
            model_save_name: str = 'None',
            lr: float = 0.001,
            batch_size: int = 64,
            epochs: int = 30,
            early_stop: int = 20,
            in_channels: int = 1,
            out_channels: int = 4,
            device: str = 'cuda',
            train_acc: str = 'Train acc',
            val_acc: str = 'Val acc',
            train_loss: str = 'Train loss',
            val_loss: str = 'Val loss',
    ):
        self.model_save_name = model_save_name
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        self.train_acc = train_acc
        self.val_acc = val_acc
        self.train_loss = train_loss
        self.val_loss = val_loss
