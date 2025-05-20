# import pandas as pd
# import numpy as np
# from torchvision import transforms, datasets
# from torch.utils.data import Dataset
# from PIL import Image
# import cv2
# import torchvision
# import os
# from script import Script
# from torch.utils.data import DataLoader
# import shutil

# script = Script()

# class seg_dataset():
#     def __init__(self, data_path, data_class, label_file_name, transform):
#         self.transform = transform
#         self.data_path = data_path
#         self.data_class = data_class
#         self.label_file_name = label_file_name

#         self.file_path = os.path.join(self.data_path, self.data_class)
#         label_path = os.path.join(self.file_path, self.label_file_name)
#         dfs = pd.read_excel(label_path, engine='openpyxl')
#         # 删除 Pterygium 为 0 的行
#         dfs = dfs[dfs['Pterygium'] != 0]
#         # 将剩下的 Image 按 list 返回
#         index_arry = dfs['Image'].tolist()
#         # 转换为字符串
#         index_arr_str = np.char.mod("%d", index_arry)

#         # 计算字符串长度
#         lengths = np.char.str_len(index_arr_str)

#         # 根据长度添加前缀
#         self.image_list = np.where(lengths == 1, np.char.add("000", index_arr_str),
#                               np.where(lengths == 2, np.char.add("00", index_arr_str),
#                                        np.where(lengths == 3, np.char.add("0", index_arr_str), index_arr_str)))


#     def __len__(self):
#         return len(self.image_list)

#     def __getitem__(self, item):
#         item =  self.image_list[item]
#         file_path = os.path.join(self.file_path, item)
#         image_name = item + '.png'
#         lael_name = item + '_label.png'
#         image_path = os.path.join(file_path, image_name)
#         label_path = os.path.join(file_path, lael_name)
#         # print(image_path)
#         # # 确保目标目录存在
#         # if not os.path.exists('data/image'):
#         #     os.makedirs('data/image')
#         # if not os.path.exists('data/mask'):
#         #     os.makedirs('data/mask')
#         # shutil.copy(image_path, 'data/image')
#         # shutil.copy(label_path, 'data/mask')

#         image_list = script.crop_image_with_overlap(image_path)
#         label_list = script.crop_image_with_overlap(label_path, islabel=True)

#         return image_list, label_list

# if __name__ == '__main__':
#     data_path = "../../../data/"
#     data_class ="test"
#     train_label_path ="test_classification_label.xlsx"
#     train_dataset = seg_dataset(data_path, data_class, train_label_path)
#     train_loader = DataLoader(dataset=train_dataset, batch_size=1, shuffle=False)

#     for image, label in train_loader:
#         pass
#         # for i in range(len(image)):
#         #     torchvision.utils.save_image(image[i], ('image/' + str(i) + '.png'))
#         #     torchvision.utils.save_image((label[i].unsqueeze(1)).float(), ('image/' + str(i) + '_label.png'))
