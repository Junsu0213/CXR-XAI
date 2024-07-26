# -*- coding:utf-8 -*-
"""
Created on Fri. Jul. 26 15:30:00 2024
@author: JUN-SU Park
"""
import os
import glob
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split


class ChestXRayDataset(Dataset):
    def __init__(self, data_config, transform=None):
        self.data_config = data_config
        self.transform = transform
        self.data_path = data_config.data_path
        self.label_list = data_config.label_list
        self.label_map = data_config.label_map
        self.data_select = data_config.data_select

        self.images = []
        self.labels = []

        # Load image paths and labels
        for label in self.label_list:
            image_path = os.path.join(self.data_path, label, self.data_select, '*.png')
            image_files = sorted(glob.glob(image_path))
            self.images.extend(image_files)
            self.labels.extend([self.label_list.index(label)] * len(image_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load and preprocess image
        img_path = self.images[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert to RGB for compatibility with PyTorch transforms
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_k_fold_loaders(self, n_splits=5, batch_size=32, random_state=42):
        # Create K-fold cross-validation data loaders
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_data_loaders = []

        for fold, (train_indices, val_indices) in enumerate(kfold.split(self)):
            train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

            train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(self, batch_size=batch_size, sampler=val_sampler)

            fold_data_loaders.append((train_loader, val_loader))

        return fold_data_loaders

    def get_train_val_test_loaders(self, test_size=0.2, val_size=0.2, batch_size=32, random_state=42):
        # Create train/validation/test data loaders
        train_val_indices, test_indices = train_test_split(
            range(len(self)), test_size=test_size, random_state=random_state,
            stratify=self.labels
        )
        train_indices, val_indices = train_test_split(
            train_val_indices, test_size=val_size / (1 - test_size), random_state=random_state,
            stratify=[self.labels[i] for i in train_val_indices]
        )

        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = DataLoader(self, batch_size=batch_size, sampler=train_sampler)
        val_loader = DataLoader(self, batch_size=batch_size, sampler=val_sampler)
        test_loader = DataLoader(self, batch_size=batch_size, sampler=test_sampler)

        return train_loader, val_loader, test_loader


def get_data_loaders(data_config, split_method='k_fold', augment=True, **kwargs):
    # Basic transformation (used when augment=False)
    basic_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Augmented transformation (used when augment=True)
    augment_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Choose appropriate transform based on augment parameter
    transform = augment_transform if augment else basic_transform

    # Create dataset
    dataset = ChestXRayDataset(data_config, transform=transform)

    # Return data loaders based on split method
    if split_method == 'k_fold':
        return dataset.get_k_fold_loaders(n_splits=kwargs.get('n_splits', 5),
                                          batch_size=kwargs.get('batch_size', 32),
                                          random_state=kwargs.get('random_state', 42))
    elif split_method == 'train_val_test':
        return dataset.get_train_val_test_loaders(test_size=kwargs.get('test_size', 0.2),
                                                  val_size=kwargs.get('val_size', 0.2),
                                                  batch_size=kwargs.get('batch_size', 32),
                                                  random_state=kwargs.get('random_state', 42))
    else:
        raise ValueError("Invalid split_method. Choose 'k_fold' or 'train_val_test'.")


def visualize_sample_data(dataloader, num_samples=5):
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        axes[i].imshow(images[i].permute(1, 2, 0))
        axes[i].set_title(f"Label: {labels[i]}")
        axes[i].axis('off')
    plt.show()


# Usage example
if __name__ == "__main__":
    from config.data_config import Covid19RadiographyDataConfig

    config = Covid19RadiographyDataConfig()

    # K-fold cross validation with augmentation
    fold_loaders = get_data_loaders(config, split_method='k_fold', augment=True, n_splits=5)
    for i, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"Fold {i + 1}:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")

    # Train/Val/Test split with augmentation
    train_loader, val_loader, test_loader = get_data_loaders(config, split_method='train_val_test', augment=True)
    print("\nTrain/Val/Test Split:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")