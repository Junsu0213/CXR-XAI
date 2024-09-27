# -*- coding:utf-8 -*-
from figure_plot.grad_cam_plot import visualize_gradcam
from config.data_config import Covid19RadiographyDataConfig
from config.model_config import ModelTrainerConfig
from model.vgg_cbam_model import VGG19
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

filter_method = 'CLAHE'
model_name = 'VGG19'
model_save_name = f'{model_name}({filter_method})'

data_config = Covid19RadiographyDataConfig()
model_config = ModelTrainerConfig(device=str(device), model_save_name=model_save_name)

# best_model_path = f"./results/model_save/{model_save_name}_best_model.pt"
best_model_path = f"./model/best_model.pt"
model = VGG19(in_channels=model_config.in_channels, out_channels=model_config.out_channels).to(model_config.device)
model.load_state_dict(torch.load(best_model_path, weights_only=True, map_location=model_config.device))
model.to(model_config.device)

for i in range(4):
    target_class = i
    file_name = '1000'

    disease_name = data_config.label_list[target_class]
    path = f'/mnt/nasw337n2/junsu_work/DATASET/COVID-CXR/COVID-19_Radiography_Dataset/{disease_name}'
    image_path = f"{path}/images/{disease_name}-{file_name}.png"
    mask_path = f"{path}/masks/{disease_name}-{file_name}.png"

    target_layer = model.conv_block5[-2]  # CBAM 이전의 마지막 컨볼루션 레이어

    visualize_gradcam(model, model_config, data_config, image_path, mask_path, target_class, target_layer)
