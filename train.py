import os
import torch
import torch.nn as nn
import torchvision.models as models
from jinja2.optimizer import optimize
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dataset import class_dataset
import pandas as pd
from model import customModel
import glob

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def train(epochs, net, batch_size, model_name):
    data_path = "../../data/"
    data_train = "train"
    data_test = "test"
    train_label_path = "train_classification_label.xlsx"
    test_label_path = "test_classification_label.xlsx"
    acc_max = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = class_dataset(data_path, data_train, train_label_path)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = class_dataset(data_path, data_test, test_label_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    net = net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        torch.cuda.empty_cache()
        net.train()
        loss_all = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            # 前向传播
            outputs = net(images)
            loss = criterion(outputs, labels)
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_all += loss.item()
            # loss_vag = loss.item() / len(images)
            # print(loss_vag)
        loss_all_avg = loss_all / len(train_dataset)
        loss_csv = model_name + "_loss.csv"
        dataframe = pd.DataFrame({'epoch': [epoch + 1], 'loss': [loss_all_avg]})
        if os.path.exists(loss_csv):
            dataframe.to_csv(loss_csv, mode='a', header=False, index=False)
        else:
            dataframe.to_csv(loss_csv, index=False)

        net.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = net(images)  # Forward pass
                _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
                correct += (predicted == labels).sum().item()
                # print(predicted)
                # print(labels)
            accuracy = correct / len(test_dataset)
            if acc_max < accuracy:
                acc_max = accuracy
                torch.save(net, "{}_max_acc.pth".format(model_name))
            print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == '__main__':

    epochs = 60
    net = customModel()
    batch_size = 32
    model_name = "VGG16"
    train(epochs=epochs, net=net, batch_size=batch_size, model_name=model_name)