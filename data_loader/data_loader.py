# -*- coding:utf-8 -*-
"""
Created on Wed. Aug. 28 14:59:21 2024
@author: JUN-SU PARK
"""
import os
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold, train_test_split


class ChestXRayDataset(Dataset):
    def __init__(self, data_config, transform=None, apply_mask=True, apply_filter=True):
        self.data_config = data_config
        self.transform = transform
        self.apply_mask = apply_mask
        self.apply_filter = apply_filter
        self.data_path = data_config.data_path
        self.label_list = data_config.label_list
        self.label_map = data_config.label_map
        self.resize = data_config.resize
        self.filter_config = data_config.filter_config

        self.images = []
        self.masks = []
        self.labels = []

        # Load image paths, mask paths, and labels
        for label in self.label_list:
            image_path = os.path.join(self.data_path, label, 'images', '*.png')
            mask_path = os.path.join(self.data_path, label, 'masks', '*.png')
            image_files = sorted(glob.glob(image_path))
            mask_files = sorted(glob.glob(mask_path))

            self.images.extend(image_files)
            self.masks.extend(mask_files)
            self.labels.extend([self.label_list.index(label)] * len(image_files))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        label = self.labels[idx]

        # Apply filtering
        if self.apply_filter and self.filter_config is not None:
            image = self.filtering(image, **self.filter_config)

        # Resize
        if self.resize is not None:
            image = cv2.resize(image, (self.resize, self.resize))
            mask = cv2.resize(mask, (self.resize, self.resize))

        # Apply lung segmentation
        if self.apply_mask:
            image = np.where(mask > 0, image, 0)

        # Covert to tensor
        image = torch.from_numpy(image).float().unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, label

    @staticmethod
    def filtering(image, method, kernel_size=5, sigma=1):
        if method == 'AF':  # averaging filter (BOX filter)
            return cv2.blur(image, (kernel_size, kernel_size))
        elif method == 'GF':  # gaussian filter
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == 'MF':  # median filter
            return cv2.medianBlur(image, kernel_size)
        elif method == 'BF':  # bilateral filter
            return cv2.bilateralFilter(image, kernel_size, sigma, sigma)
        elif method == 'LF':  # laplacian filter
            return cv2.Laplacian(image, -1)
        elif method == 'Canny':  # Canny filter
            return cv2.Canny(image, 100, 200)
        elif method == 'HE':  # histogram equalization
            return cv2.equalizeHist(image)
        elif method == 'CLAHE':  # contrast limited adaptive HE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
        else:
            return image

    def get_k_fold_loaders(self, n_splits=5, batch_size=32, random_state=42):
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


def get_integrated_data_loaders(data_config, split_method='k_fold', augment=True, **kwargs):
    # Basic transformation
    basic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Augmented transformation
    augment_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    transform = augment_transform if augment else basic_transform

    dataset = ChestXRayDataset(data_config, transform=transform)

    if split_method == 'k_fold':
        return dataset.get_k_fold_loaders(n_splits=kwargs.get('n_splits', 10),
                                          batch_size=kwargs.get('batch_size', 32),
                                          random_state=kwargs.get('random_state', 42))
    elif split_method == 'train_val_test':
        return dataset.get_train_val_test_loaders(test_size=kwargs.get('test_size', 0.2),
                                                  val_size=kwargs.get('val_size', 0.2),
                                                  batch_size=kwargs.get('batch_size', 32),
                                                  random_state=kwargs.get('random_state', 42))
    else:
        raise ValueError("Invalid split_method. Choose 'k_fold' or 'train_val_test'.")


# Usage example
if __name__ == "__main__":
    from config.data_config import Covid19RadiographyDataConfig
    import matplotlib.pyplot as plt

    config = Covid19RadiographyDataConfig()

    # K-fold cross validation with augmentation
    fold_loaders = get_integrated_data_loaders(config, split_method='k_fold', augment=True, n_splits=10)
    for i, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"Fold {i + 1}:")
        print(f"  Training batches: {len(train_loader)}")
        print(f"  Validation batches: {len(val_loader)}")


    # Train/Val/Test split with augmentation
    train_loader, val_loader, test_loader = get_integrated_data_loaders(config, split_method='train_val_test',
                                                                        augment=True)
    print("\nTrain/Val/Test Split:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")


    # Visualize sample data
    def visualize_sample_data(dataloader, num_samples=5):
        images, labels = next(iter(dataloader))
        print(images[0].shape)
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].set_title(f"Label: {config.label_list[labels[i]]}, Processed: {config.filter_config['method']}")
            axes[i].axis('off')
        plt.show()


    visualize_sample_data(test_loader)
