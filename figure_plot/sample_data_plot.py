# -*- coding:utf-8 -*-
"""
Created on Sun. Sep. 29 00:49:45 2024
@author: JUN-SU PARK
"""
import matplotlib.pyplot as plt


# Visualize sample data
def visualize_sample_data(data_config, dataloader, num_samples=5):
    images, labels = next(iter(dataloader))
    print(images[0].shape)
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {data_config.label_list[labels[i]]}, Processed: {data_config.filter_config['method']}")
        axes[i].axis('off')
    plt.show()
