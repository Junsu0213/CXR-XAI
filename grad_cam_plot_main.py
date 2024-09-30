# -*- coding:utf-8 -*-
from figure_plot.grad_cam_plot import visualize_gradcam
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from model.vgg_cbam_model import VGG19
import wandb
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filter_list = ['CLAHE', 'MF'] # 'LF', 'Canny', 'MF', 'Origin', 'CLAHE','HE',

for filter_method in filter_list:

    model_name = 'VGG19'
    model_save_name = f'{model_name}({filter_method})'

    data_config = Covid19RadiographyDataConfig()
    data_config.filter_config['method'] = filter_method

    model_config = ModelTrainerConfig(device=str(device), model_save_name=model_save_name)

    best_model_path = f"./results/model_save/{model_save_name}_best_model.pt"
    # best_model_path = f"./model/best_model.pt"
    model = VGG19(in_channels=model_config.in_channels, out_channels=model_config.out_channels).to(model_config.device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=model_config.device))
    model.to(model_config.device)

    for i in range(4):
        target_class = i

        disease_name = data_config.label_list[target_class]
        path = f'/mnt/nasw337n2/junsu_work/DATASET/COVID-CXR/COVID-19_Radiography_Dataset/grad_cam'
        image_path = f"{path}/images/{disease_name}.png"
        mask_path = f"{path}/masks/{disease_name}.png"

        target_layer = model.conv_block5[-2]  # CBAM 이전의 마지막 컨볼루션 레이어

        wandb.init(
            project='CXR-XAI',
            name=f'{model_save_name}_grad_cam_{disease_name}',
            config={
                'learning_rate': model_config.lr,
                'batch_size': model_config.batch_size,
                'epochs': model_config.epochs,
                'model': f'{model_name}_CBAM',
                'dataset': 'COVID-19_Radiography_Dataset',
                'filter': filter_method,
            }
        )

        visualize_gradcam(model, model_config, data_config, image_path, mask_path, target_class, target_layer)

        wandb.finish()
