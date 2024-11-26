# -*- coding:utf-8 -*-
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from trainer.train import ModelTrainer
from data_loader.data_loader import get_integrated_data_loaders
from model.vgg_cbam_model import VGG19
from model.focal_loss import FocalLoss
from figure_plot.con_matrix_plot import plot_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score
from tabulate import tabulate
import numpy as np
import wandb
import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Available GPUs: {torch.cuda.device_count()}")

filter_list = ['HE']  # , 'GF', 'MF', 'LF', 'Canny', 'Origin', 'CLAHE'
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
        train_acc=f'Train acc (Kfold, {filter_method})_ver.2',
        val_acc=f'Val acc (Kfold, {filter_method})_ver.2',
        train_loss=f'Train loss (Kfold, {filter_method})_ver.2',
        val_loss=f'Val loss (Kfold, {filter_method})_ver.2',
    )

    all_folds_preds = []
    all_folds_labels = []
    all_folds_acc = []
    all_folds_f1_score = []

    fold_results = []

    # K-fold cross validation with augmentation
    fold_loaders = get_integrated_data_loaders(
        data_config,
        split_method='k_fold',
        augment=True,
        batch_size=model_config.batch_size * torch.cuda.device_count(),  # 배치 크기를 GPU 개수만큼 증가
        n_splits=5,
    )
    for i, (train_loader, val_loader) in enumerate(fold_loaders):

        model = VGG19(in_channels=model_config.in_channels, out_channels=model_config.out_channels)
        model = nn.DataParallel(model)  # 여러 GPU에서 모델을 병렬 처리
        model = model.to(device)
        criterion = FocalLoss(alpha=alpha)
        model_trainer = ModelTrainer(model_config=model_config, model=model, criterion=criterion)

        wandb.init(
            project='CXR-XAI',
            name=f'{model_save_name}_fold_{i + 1}_ver.2',
            config={
                'learning_rate': model_config.lr,
                'batch_size': model_config.batch_size * torch.cuda.device_count(),
                'epochs': model_config.epochs,
                'model': f'{model_name}_CBAM',
                'dataset': 'COVID-19_Radiography_Dataset',
                'filter': filter_method,
                'loss': loss,
                'Focal Loss Alpha': alpha,
                'num_gpus': torch.cuda.device_count(),
            }
        )

        print(f"Fold {i + 1}:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

        model_trainer.train(train_loader=train_loader)

        all_preds = []
        all_labels = []

        model.eval()  # 평가 모드로 설정
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(device)
                outputs = model(X)
                _, pred = torch.max(outputs, 1)
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(y.numpy())

        all_folds_preds.extend(all_preds)
        all_folds_labels.extend(all_labels)

        acc_ = accuracy_score(np.array(all_labels), np.array(all_preds))
        f1_score_ = f1_score(np.array(all_labels), np.array(all_preds), average='macro')

        all_folds_acc.append(acc_)
        all_folds_f1_score.append(f1_score_)

        fold_results.append([f"Fold {i + 1}", f"{acc_:.4f}", f"{f1_score_:.4f}"])

        wandb.finish()

    all_folds_preds = np.array(all_folds_preds)
    all_folds_labels = np.array(all_folds_labels)

    # Confusion Matrix 시각화
    plot_confusion_matrix(all_folds_labels, all_folds_preds, data_config.label_list, file_name=model_save_name)

    # 결과 출력
    print(f"\nResults for {filter_method}:")
    print(tabulate(fold_results, headers=["Fold", "Accuracy", "F1 Score"], tablefmt="grid"))

    # 평균 및 표준편차 계산
    mean_acc = np.mean(all_folds_acc)
    std_acc = np.std(all_folds_acc)
    mean_f1 = np.mean(all_folds_f1_score)
    std_f1 = np.std(all_folds_f1_score)

    print(f"\nFinal Results for {filter_method}:")
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro F1 Score: {mean_f1:.4f} ± {std_f1:.4f}")
