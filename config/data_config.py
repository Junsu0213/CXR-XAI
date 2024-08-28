# -*- coding:utf-8 -*-
"""
Created on Fri. Jul. 05 14:22:31 2024
@author: JUN-SU Park
"""


class Covid19RadiographyDataConfig:
    """
    images size: 299*299 pixel
    lung_masks size: 256*256 pixel
    """
    def __init__(
            self,
            data_path='/mnt/nasw337n2/junsu_work/DATASET/COVID-CXR/COVID-19_Radiography_Dataset',
            label_list=None,
            label_map=None,
            data_select='segmented_images',
            resize=256,
            image_filtering=True,
            filter_config=None,
    ):

        if label_list is None:
            label_list = ['Normal', 'COVID', 'Lung_Opacity', 'Viral_Pneumonia']
        if label_map is None:
            label_map = {0: 'Normal', 1: 'COVID', 2: 'Lung_Opacity', 3: 'Viral Pneumonia'}
        if image_filtering is True:
            filter_config = {'method': 'HE', 'kernel_size': 5, 'sigma': 1}
        '''
        Filtering Method Abbreviations

        ** Blurring **
        AF: Averaging filter
        GF: Gaussian filter
        MF: Median filter
        BF: Bilateral filter

        ** Edge detection **
        LF: Laplacian filter
        Canny: Canny filter

        ** Histogram Equalization **
        HE: Histogram equalization
        CLAHE: Contrast limited adaptive histogram equalization
        '''

        self.data_path = data_path
        self.label_list = label_list
        self.label_map = label_map
        self.data_select = data_select
        self.resize = resize
        self.image_filtering = image_filtering
        self.filter_config = filter_config


class CovidQuExDataConfig:
    """
    images size: 256*256 pixel
    lung_masks size: 256*256 pixel
    """

    def __init__(
            self,
            data_path=r'/home/wlsdud022/autogluon_data/JS_DATASET/COVID/COVID-QU-Ex_Dataset',
            label_list=None,
            label_map=None,
            data_select='segmented_images',
            resize=None,
            image_filtering=False,
            filter_config=None
    ):

        if label_list is None:
            label_list = ['Normal', 'COVID', 'Non_Covid_Pneumonia']
        if label_map is None:
            label_map = {0: 'Normal', 1: 'COVID', 2: 'Non_Covid_Pneumonia'}
        if image_filtering is True:
            filter_config = {'method': 'CLAHE', 'kernel_size': 5, 'sigma': 1}
        '''
        Filtering Method Abbreviations

        ** Blurring **
        AF: Averaging filter
        GF: Gaussian filter
        MF: Median filter
        BF: Bilateral filter

        ** Edge detection **
        LF: Laplacian filter

        ** Histogram **
        HE: Histogram equalization
        CLAHE: Contrast limited adaptive histogram equalization
        '''

        self.data_path = data_path
        self.label_list = label_list
        self.label_map = label_map
        self.data_select = data_select
        self.resize = resize
        self.image_filtering = image_filtering
        self.filter_config = filter_config
