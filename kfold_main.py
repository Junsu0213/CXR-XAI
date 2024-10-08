# -*- coding:utf-8 -*-
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from trainer.train import ModelTrainer
from data_loader.data_loader import get_integrated_data_loaders
from model.vgg_cbam_model import VGG19
from model.focal_loss import FocalLoss
from figure_plot.con_matrix_plot import plot_confusion_matrix
import numpy as np
import wandb
import torch
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


filter_list = ['HE'] #, 'GF', 'MF', 'LF', 'Canny', 'Origin', 'CLAHE'
loss = 'focal_loss'
alpha = [0.2, 0.4, 0.3, 0.1]

for filter_method in filter_list:

    model_name = 'VGG19_Kfold'
    model_save_name = f'{model_name}({filter_method})'

    data_config = Covid19RadiographyDataConfig()
    data_config.filter_config['method'] = filter_method

    model_config = ModelTrainerConfig(
        device=str(device),
        model_save_name=model_save_name,
        train_acc=f'Train acc (Kfold, {filter_method})',
        val_acc=f'Val acc (Kfold, {filter_method})',
        train_loss=f'Train loss (Kfold, {filter_method})',
        val_loss=f'Val loss (Kfold, {filter_method})',
    )

    model = VGG19(in_channels=model_config.in_channels, out_channels=model_config.out_channels).to(device)
    model_trainer = ModelTrainer(model_config=model_config, model=model)

    all_folds_preds = []
    all_folds_labels = []

    # K-fold cross validation with augmentation
    fold_loaders = get_integrated_data_loaders(data_config=data_config, model_config=model_config, split_method='k_fold', augment=True)
    for i, (train_loader, val_loader) in enumerate(fold_loaders):

        wandb.init(
            project='CXR-XAI',
            name=f'{model_save_name}_fold_{i + 1}',
            config={
                'learning_rate': model_config.lr,
                'batch_size': model_config.batch_size,
                'epochs': model_config.epochs,
                'model': f'{model_name}_CBAM',
                'dataset': 'COVID-19_Radiography_Dataset',
                'filter': filter_method,
                'loss': loss,
                'Focal Loss Alpha': alpha,
            }
        )

        print(f"Fold {i + 1}:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

        model_trainer.train(train_loader=train_loader)

        all_preds = []
        all_labels = []

        for X, y in val_loader:
            pred, _ = model_trainer.predicate(X)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.numpy())

        all_folds_preds.extend(all_preds)
        all_folds_labels.extend(all_labels)

    all_folds_preds = np.array(all_folds_preds)
    all_folds_labels = np.array(all_folds_labels)

    # Confusion Matrix 시각화
    plot_confusion_matrix(all_folds_labels, all_folds_preds, data_config.label_list, file_name=model_save_name)
