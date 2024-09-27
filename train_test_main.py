# -*- coding:utf-8 -*-
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from trainer.train import ModelTrainer
from data_loader.data_loader import get_integrated_data_loaders
from model.vgg_cbam_model import VGG19
from figure_plot.con_matrix_plot import plot_confusion_matrix
import numpy as np
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filter_method = 'CLAHE'
model_name = 'VGG19'
model_save_name = f'{model_name}({filter_method})'

data_config = Covid19RadiographyDataConfig()
data_config.filter_config['method'] = filter_method

model_config = ModelTrainerConfig(device=str(device), model_save_name=model_save_name)

model = VGG19(in_channels=model_config.in_channels, out_channels=model_config.out_channels).to(device)
model_trainer = ModelTrainer(model_config=model_config, model=model)

train_loader, val_loader, test_loader = get_integrated_data_loaders(
    data_config,
    split_method='train_val_test',
    agment=True
)

print("\nTrain/Val/Test Split:")
print(f"Training batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

model_trainer.train(train_loader, val_loader)

all_preds = []
all_labels = []

for X, y in test_loader:
    pred, _ = model_trainer.predicate(X)
    all_preds.extend(pred.cpu().numpy())
    all_labels.extend(y.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion Matrix 시각화
plot_confusion_matrix(all_labels, all_preds, data_config.label_list, file_name=model_save_name)
