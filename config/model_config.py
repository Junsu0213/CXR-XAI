# -*- coding:utf-8 -*-
"""
Created on Thu. Sep. 26 10:07:11 2024
@author: JUN-SU Park
"""


class ModelTrainerConfig:
    def __init__(
            self,
            lr: float = 0.001,
            batch_size: int = 32,
            epochs: int = 100,
            early_stop: int = 80,
            in_channels: int = 1,
            num_classes: int = 4,
            device: str = 'cuda'
    ):
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stop = early_stop
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.device = device
