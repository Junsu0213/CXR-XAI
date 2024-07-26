# -*- coding:utf-8 -*-
"""
Created on Fri. Jul. 05 14:50:10 2024
@author: JUN-SU Park
"""
import os
import glob
import numpy as np
import cv2
from matplotlib import pyplot as plt
from config.data_config import CovidQuExDataConfig, Covid19RadiographyDataConfig


class DataPreprocessing:
    def __init__(self, data_config):
        self.data_config = data_config
        self.data_path = data_config.data_path
        self.label_list = data_config.label_list
        self.label_map = data_config.label_map
        self.resize = data_config.resize
        self.filter_config = data_config.filter_config

    def lung_segmentation(self):

        for label in self.label_list:
            segmented_image_save_path = os.path.join(rf'{self.data_path}/{label}/segmented_images')

            # Create if the folder does not exit
            os.makedirs(segmented_image_save_path, exist_ok=True)

            # Path for image and lung mask
            images_path = os.path.join(rf'{self.data_path}/{label}/images/*.png')
            lung_masks_path = os.path.join(rf'{self.data_path}/{label}/lung_masks/*.png')

            # List of image and lung mask files
            images_files = sorted(glob.glob(images_path))
            lung_masks_files = sorted(glob.glob(lung_masks_path))

            # Check the number of data
            num_images = len(images_files)
            print(f'Number of {label} images: {num_images}')

            for image_file, lung_mask_file in zip(images_files, lung_masks_files):
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
                lung_mask_image = cv2.imread(lung_mask_file, cv2.IMREAD_GRAYSCALE)

                # Filtering
                if self.filter_config is not None:
                    image = self.filtering(image=image, method=self.filter_config['method'],
                                           kernel_size=self.filter_config['kernel_size'], sigma=self.filter_config['sigma'])

                # Resize
                if self.resize is not None:
                    image = cv2.resize(image, (self.resize, self.resize))
                    lung_mask_image = cv2.resize(lung_mask_image, (self.resize, self.resize))

                # Lung segmentation
                segmented_image = np.where(lung_mask_image > 0, image, 0)

                # Set the files path for saving
                base_filename = os.path.basename(image_file)
                save_path = os.path.join(segmented_image_save_path, base_filename)

                # Save the segmented image
                cv2.imwrite(save_path, segmented_image)

                # # segmented image plot
                # plt.imshow(segmented_image, cmap='gray')
                # plt.title(f'Masked Image {base_filename}')
                # plt.show()

    @staticmethod
    def filtering(image, method, kernel_size=5, sigma=1):

        # Noise Reduction (Blur)
        if method == 'AF':  # averaging filter (BOX filter)
            return cv2.blur(image, (kernel_size, kernel_size))
        elif method == 'GF':  # gaussian filter
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        elif method == 'MF':  # median filter
            return cv2.medianBlur(image, kernel_size)
        elif method == 'BF':  # bilateral filter
            return cv2.bilateralFilter(image, kernel_size)

        # Image Gradients (Edge Detection)
        elif method == 'LF':  # laplacian filter
            # print(f'Input image type: {image.dtype}')
            return cv2.Laplacian(image, -1)
        elif method == 'Canny':  # Canny filter
            return cv2.Canny(image, 100, 200)

        # Histogram Equalization (Enhance Contrast)
        elif method == 'HE':  # histogram_equalization (HE)
            return cv2.equalizeHist(image)
        elif method == 'CLAHE':  # contrast limited adaptive HE (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)


if __name__ == '__main__':
    # COVID-19_Radiography_Dataset Configuration
    config = Covid19RadiographyDataConfig()

    # # COVID-QU-Ex_Dataset Configuration
    # config = CovidQuExDataConfig()

    preprocessing = DataPreprocessing(config)
    preprocessing.lung_segmentation()
