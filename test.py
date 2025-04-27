import os
import torch
import pandas as pd
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    # transforms.ToPILImage(),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(15),
    # transforms.Normalize( [0.4914, 0.4822, 0.4465],  [0.2023, 0.1994, 0.2010])
])

class class_dataset(Dataset):
    def __init__(self, data_path, istransform=True):
        self.istransform = istransform
        self.data_path = data_path
        self.image_list = os.listdir(self.data_path)

    def __getitem__(self, index):
        file_name = self.image_list[index]
        image_path = os.path.join(self.data_path, file_name)
        img = Image.open(image_path)
        img = np.array(img)
        if self.istransform :
            img = transform(img)

        return img, file_name

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "../../data/val_img/"

    val_dataset = class_dataset(data_path)

    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = torch.load('VGG16_max_acc_0.9555555555555556.pth', weights_only=False)
    model = model.to(device)

    with torch.no_grad():
        correct = 0
        for images, name in valid_loader:
            images = images.to(device)
            outputs = model(images)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get the class with the highest probability
            print(predicted.item(), name[0])