# -*- coding:utf-8 -*-
"""
Created on Mon. Sep. 23 10:23:01 2024
@author: JUN-SU Park
"""
import wandb
import torch
import torch.nn as nn
import torchmetrics
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class ModelTrainer:
    def __init__(
            self,
            model_config,
            model,
            criterion=None,
            optimizer=None,
            scheduler=None
    ):
        if criterion is None:
            criterion = nn.CrossEntropyLoss().to(model_config.device)
        if optimizer is None:
            optimizer = torch.optim.Adam(model.parameters(), lr=model_config.lr, betas=(0.9, 0.98), eps=1e-9)
        if scheduler is None:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        self.model_config = model_config
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self, train_loader, val_loader=None):
        if val_loader is None:
            full_dataset = train_loader.dataset
            dataset_size = len(full_dataset)
            train_size = int(0.8*dataset_size)
            val_size = dataset_size - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.model_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.model_config.batch_size, shuffle=True)

        metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.model_config.out_channels).to(
            self.model_config.device)

        loss_list = []
        early_stop_counter = 0

        for epoch in range(1, self.model_config.epochs + 1):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for x, y in train_loader:
                x = x.to(self.model_config.device)
                y = y.to(self.model_config.device)
                self.optimizer.zero_grad()  # 그래디언트 초기화
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                pred_ = pred.cpu()
                y_ = y.cpu()
                acc = metric(pred_, y_)
                train_loss += loss.item()
                train_acc += acc

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            self.model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.model_config.device)
                    y = y.to(self.model_config.device)
                    pred = self.model(x)
                    loss = self.criterion(pred, y)
                    pred_ = pred.cpu()
                    y_ = y.cpu()
                    acc_ = metric(pred_, y_)
                    val_loss += loss.item()
                    val_acc += acc_

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            wandb.log({
                self.model_config.train_acc: train_acc,
                self.model_config.train_loss: train_loss,
                self.model_config.val_acc: val_acc,
                self.model_config.val_loss: val_loss
            })

            if epoch % 10 == 0:
                print(
                    f'Epoch: {epoch}\n'
                    f'Train Accuracy: {train_acc:.4f}, Train Loss: {train_loss:.4f}\n'
                    f'Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}'
                )
                print('############################################################################')

            loss_list.append(val_loss)
            best_loss = min(loss_list)

            if val_loss > best_loss:
                early_stop_counter += 1
            else:
                early_stop_counter = 0
                torch.save(self.model.state_dict(), f'./results/model_save/{self.model_config.model_save_name}_best_model.pt')

            if early_stop_counter >= self.model_config.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step()  # 학습률 스케줄러 업데이트

        print("Training completed.")

    def predicate(self, X):
        self.model.load_state_dict(torch.load(f'./results/model_save/{self.model_config.model_save_name}_best_model.pt', weights_only=True, map_location=self.model_config.device))
        self.model.to(self.model_config.device)
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.model_config.device)
            out = self.model(X)
            out_ = out.cpu()
            prob = F.softmax(out_, dim=1)
            pred = torch.argmax(prob, dim=1)
        return pred, prob
