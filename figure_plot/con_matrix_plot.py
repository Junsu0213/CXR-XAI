# -*- coding:utf-8 -*-
"""
Created on Fri. Sep. 27 10:52:30 2024
@author: JUN-SU PARK
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score


def plot_confusion_matrix(y_true, y_pred, classes, file_name):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 확률로 정규화

    fig, ax = plt.subplots(figsize=(16, 15))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')

    # 글꼴 크기 설정
    MEDIUM_SIZE = 25
    BIGGER_SIZE = 30

    # 확률과 개수를 함께 표시
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_norm[i, j]:.2f}\n({cm[i, j]})"
            ax.text(j, i, text, ha="center", va="center", color="black" if cm_norm[i, j] < 0.5 else "white", fontsize=BIGGER_SIZE)

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, fontsize=MEDIUM_SIZE)
    ax.set_yticklabels(classes, fontsize=MEDIUM_SIZE)

    plt.setp(ax.get_yticklabels(), rotation=90, va='center')

    ax.set_title('Confusion Matrix', fontsize=BIGGER_SIZE, pad=20)
    ax.set_ylabel('True Label', fontsize=BIGGER_SIZE, rotation=90, va='center', labelpad=20)
    ax.set_xlabel('Predicted Label', fontsize=BIGGER_SIZE, labelpad=20)

    # Accuracy와 F1 Score 계산
    accuracy = accuracy_score(y_true, y_pred)
    f1_avg = f1_score(y_true, y_pred, average='macro')

    # Accuracy와 Average F1 Score를 한 줄에 표시
    plt.text(0.5, -0.15, f"Accuracy: {accuracy:.4f}     Average F1 Score: {f1_avg:.4f}",
             transform=ax.transAxes, ha='center', va='center', fontsize=MEDIUM_SIZE)

    fig.tight_layout()

    file_path = f'./results/con_matrix/{file_name}_confusion_matrix.png'
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    print(f"Confusion matrix saved to {file_path}")
