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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def visualize_gradcam(model, image_path, target_class, target_layer):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert('L')  # grayscale
    image = transforms.Resize((256, 256))(image)
    image = transforms.ToTensor()(image).unsqueeze(0)

    # Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_cam(image.to(device), target_class)

    # 원본 이미지와 히트맵 결합
    image = image.squeeze().cpu().numpy()
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + image * 0.6

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap)
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(superimposed_img)
    plt.title('Superimposed')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    from model.vgg_cbam_model import VGG19
    model = VGG19(in_channels=1, out_channels=4).to(device)
    model.load_state_dict(torch.load("./best_model.pt"))

    target_layer = model.conv_block5[-2]

    image_path = ''
    target_class = 0

    visualize_gradcam(model, image_path, target_class, target_layer)
