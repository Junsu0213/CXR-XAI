# -*- coding:utf-8 -*-
"""
Created on Mon. Sep. 23 10:23:01 2024
@author: JUN-SU Park
"""
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
            train_dataset, val_dataset = torch.utils.data.random_split(train_loader, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=self.model_config.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.model_config.batch_size, shuffle=True)

        metric = torchmetrics.Accuracy(task='multiclass', num_classes=self.model_config.num_classes).to(self.model_config.device)

        loss_list = []
        early_stop_counter = 0

        for epoch in range(self.model_config.epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for x, y in train_loader:
                x = x.to(self.model_config.device)
                y = y.to(self.model_config.device)
                pred = self.model(x)
                loss = self.criterion(pred, y)
                loss.backward()
                self.optimizer.step()
                pred_ = pred.cpu()
                y_ = y.cpu()
                acc = metric(pred_, y_)
                train_loss += loss.item()
                train_acc += acc

                if epoch % 10 == 0:
                    print(
                        f'Epoch: {epoch}, '
                        f'Train Accuracy: {train_acc / len(train_loader): .4f},'
                        f'Train Loss: {train_loss / len(train_loader):.4f}'
                    )

                self.model.eval()
                with torch.no_grad():
                    val_loss = 0
                    val_acc = 0
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

                if epoch % 10 == 0:
                    print(
                        f'Epoch: {epoch}, '
                        f'Validation Accuracy: {val_acc / len(val_loader): .4f},'
                        f'Validation Loss: {val_loss / len(val_loader):.4f}'
                    )
                    print('############################################################################')

                v_loss = val_loss/len(val_loader)
                loss_list.append(v_loss)
                best_loss = min(loss_list)

                if v_loss > best_loss:
                    early_stop_counter += 1
                else:
                    early_stop_counter = 0
                    torch.save(self.model.state_dict(), '../model/best_model.pt')

                if early_stop_counter >= self.model_config.early_stop:
                    break

    def predicate(self, X):
        model = torch.load('../model/best_model.pt', map_location=self.model_config.device)
        with torch.no_grad():
            model.eval()
            out = model(X)
            out_ = out.cpu()
            prob = F.softmax(out_, dim=1)
            pred = torch.argmax(prob, dim=1)
        return pred, prob
