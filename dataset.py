import os
from operator import index

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
    def __init__(self, data_path, data_class, label_file_name, istransform=True):
        self.istransform = istransform
        self.data_path = data_path
        self.data_class = data_class
        self.label_file_name = label_file_name

        self.file_path = os.path.join(self.data_path, self.data_class)
        label_path = os.path.join(self.file_path, self.label_file_name)
        dfs = pd.read_excel(label_path, engine='openpyxl')

        datas_arry = dfs.to_numpy()
        index_arry = datas_arry[:, 0]
        # 转换为字符串
        index_arr_str = np.char.mod("%d", index_arry)

        # 计算字符串长度
        lengths = np.char.str_len(index_arr_str)

        # 根据长度添加前缀
        index_arry = np.where(lengths == 1, np.char.add("000", index_arr_str),
                              np.where(lengths == 2, np.char.add("00", index_arr_str),
                                       np.where(lengths == 3, np.char.add("0", index_arr_str), index_arr_str)))
        # 数据集index列表和label列表
        self.index_arry = index_arry
        self.label_arry = datas_arry[:, 1]

    def __getitem__(self, index):
        file_name = self.index_arry[index]
        file_path = os.path.join(self.file_path, file_name)
        image_path = os.path.join(file_path, (file_name + ".png"))
        img = Image.open(image_path)
        img = np.array(img)
        label = self.label_arry[index]
        if self.istransform :
            img = transform(img)
            # print(image_path)
            # img.save("test.png")

        return img, label

    def __len__(self):
        return len(self.index_arry)

if __name__ == '__main__':
    data_path = "../../data/"
    data_class ="train"
    train_label_path ="train_classification_label.xlsx"
    train_dataset = class_dataset(data_path, data_class, train_label_path)
    img, label = train_dataset.__getitem__(2)
    print(img, label)
    # print(len(train_dataset))
