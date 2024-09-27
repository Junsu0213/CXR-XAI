# -*- coding:utf-8 -*-
"""
Created on Thu. Sep. 26 10:07:11 2024
@author: JUN-SU PARK
"""


class ModelTrainerConfig:
    def __init__(
            self,
            lr: float = 0.001,
            batch_size: int = 32,
            epochs: int = 30,
            early_stop: int = 20,
            in_channels: int = 1,
            num_classes: int = 4,
            device: str = 'cuda:0'
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
