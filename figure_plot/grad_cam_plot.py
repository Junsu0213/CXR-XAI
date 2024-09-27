# -*- coding:utf-8 -*-
"""
Created on Thu. Sep. 26 16:10:20 2024
@author: JUN-SU PARK
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from data_loader.data_loader import ChestXRayDataset


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


def preprocess_image(image_path, mask_path, data_config):
    # 이미지와 마스크 로드
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # 필터링 적용
    image = ChestXRayDataset.filtering(image, **data_config.filter_config)

    # 리사이즈
    if data_config.resize is not None:
        image = cv2.resize(image, (data_config.resize, data_config.resize))
        mask = cv2.resize(mask, (data_config.resize, data_config.resize))

    # 마스크 적용
    image = np.where(mask > 0, image, 0)

    # Covert to tensor
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)

    # 텐서로 변환 및 정규화
    basic_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = basic_transform(image_tensor)

    return image_tensor.unsqueeze(0), image


def visualize_gradcam(model, model_config, data_config, image_path, mask_path, true_class, target_layer):

    image_tensor, original_image = preprocess_image(image_path, mask_path, data_config)

    # 모델 예측
    model.eval()
    with torch.no_grad():
        X = image_tensor.to(model_config.device)
        out = model(X)
        out_ = out.cpu()
        prob = torch.nn.functional.softmax(out_, dim=1)
        predicted_class = torch.argmax(prob, dim=1)

    # Grad-CAM 생성
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate_cam(image_tensor.to(model_config.device), predicted_class)

    heatmap = cv2.resize(heatmap, (data_config.resize, data_config.resize))

    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    superimposed_img = cv2.addWeighted(original_image_rgb, 0.6, heatmap_color, 0.4, 0)

    # 클래스 이름 설정
    class_names = data_config.label_list
    predicted_name = class_names[predicted_class]
    true_name = class_names[true_class]

    # 시각화
    plt.figure(figsize=(12, 4))
    plt.subplot(131)
    plt.imshow(original_image, cmap='gray')
    plt.title(f'Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(heatmap, cmap='jet')
    plt.title(f'Grad-CAM Heatmap')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.title('Superimposed')
    plt.axis('off')

    plt.suptitle(f'True: {true_name}, Predicted: {predicted_name}', fontsize=16)
    plt.tight_layout()

    file_path = f'./results/grad_cam/{model_config.model_save_name}_{true_name}.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"Grad cam saved to {file_path}")
