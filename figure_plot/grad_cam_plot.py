# -*- coding:utf-8 -*-
"""
Created on Thu. Sep. 26 16:10:20 2024
@author: JUN-SU PARK
"""
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from torchvision import transforms


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class):
        # 모델 추론
        self.model.eval()
        output = self.model(input_image)

        # 타겟 클래스에 대한 그래디언트 계산
        self.model.zero_grad()
        output[:, target_class].backward()

        # Grad-CAM 생성
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap /= torch.max(heatmap)

        return heatmap.cpu().numpy()


def visualize_gradcam(model, model_config, image_path, mask_path, target_class, target_layer):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('L')  # grayscale
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    image_tensor = transform(image).unsqueeze(0)

    # Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_cam(image_tensor.to(model_config.device), target_class)

    # 원본 이미지와 히트맵 결합
    image_np = np.array(image)
    heatmap = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 이미지를 3채널로 변환
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)

    # 이미지와 히트맵의 값 범위를 0-255로 맞춤
    image_rgb = cv2.normalize(image_rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    superimposed_img = cv2.addWeighted(image_rgb, 0.6, heatmap, 0.4, 0)

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image_np, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Superimposed')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from model.vgg_cbam_model import VGG19
    from config.model_config import ModelTrainerConfig

    device = 'cuda:1'
    model_config = ModelTrainerConfig(device=str(device))

    model = VGG19(in_channels=1, out_channels=4).to(device)
    model.load_state_dict(torch.load("../model/best_model.pt"))

    target_layer = model.conv_block5[-2]

    image_path = r'/home/wlsdud022/junsu_work/DATASET/COVID-CXR/COVID-19_Radiography_Dataset/grad_cam_test/COVID/COVID-27.png'
    target_class = 0

    visualize_gradcam(model, model_config, image_path, target_class, target_layer)
