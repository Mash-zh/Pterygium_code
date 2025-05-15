from dataset import seg_dataset


def make_box_data(originalData_path, dataWithBox_path, label_file_name, mode):
    train_dataset = seg_dataset(originalData_path, mode, label_file_name)
    for _, mask in train_dataset:
        print(mask)

if __name__ == '__main__':
    #制作box数据集
    data_path = "../../../data/"
    data_class = "train"
    train_label_path = "train_classification_label.xlsx"
    make_box_data(originalData_path=data_path, dataWithBox_path='./',
                         label_file_name=train_label_path, mode=data_class)