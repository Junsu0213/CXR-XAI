# -*- coding:utf-8 -*-
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from trainer.train import ModelTrainer
from data_loader.data_loader import get_integrated_data_loaders
from model.vgg_cbam_model import VGG19

import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

data_config = Covid19RadiographyDataConfig()
train_config = ModelTrainerConfig()

model = VGG19(in_channels=train_config.in_channels, out_channels=train_config.num_classes).to(train_config.device)
model_trainer = ModelTrainer(model_config=train_config, model=model)

# K-fold cross validation with augmentation
fold_loaders = get_integrated_data_loaders(data_config=data_config, split_method='k_fold', augment=True, n_splits=5)
for i, (train_loader, val_loader) in enumerate(fold_loaders):
    print(f"Fold {i + 1}:")
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")

    model_trainer.train(train_loader=train_loader)
