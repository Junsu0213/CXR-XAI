# -*- coding:utf-8 -*-
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from trainer.train import ModelTrainer
from data_loader.data_loader import get_integrated_data_loaders
from model.vgg_cbam_model import VGG19
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from figure_plot.con_matrix_plot import plot_confusion_matrix

import numpy as np
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
epochs = 20

data_config = Covid19RadiographyDataConfig()
train_config = ModelTrainerConfig(device=str(device), epochs=epochs)

model = VGG19(in_channels=train_config.in_channels, out_channels=train_config.num_classes).to(train_config.device)
model_trainer = ModelTrainer(model_config=train_config, model=model)

# Train/Val/Test split with augmentation
train_loader, val_loader, test_loader = get_integrated_data_loaders(data_config, split_method='train_val_test',
                                                                    augment=True)
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

acc = accuracy_score(all_labels, all_preds)
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds)

print(f"Test Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(class_report)

# Confusion Matrix 시각화
plot_confusion_matrix(all_labels, all_preds, data_config.label_list, file_name='test')
