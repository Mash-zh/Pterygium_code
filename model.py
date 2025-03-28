import torch
import torch.nn as nn
import torchvision.models as models

class customModel(nn.Module):
    def __init__(self, num_class=3, ispretrained=True):
        super().__init__()
        self.num_class = num_class
        self.ispretrained = ispretrained
        self.model = self.__net__()

    def __net__(self):
        # 加载预训练的 VGG16 模型
        model = models.vgg16(pretrained=self.ispretrained)
        # 获取输入特征数（VGG16 原始 fc 层输出 1000 类）
        num_features = model.classifier[6].in_features
        # 替换最后一层，全连接层输出 3 类
        model.classifier[6] = nn.Linear(num_features, self.num_class)
        return model
        # 打印模型，检查修改是否生效
        # print(model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = customModel()
    print(list(model.parameters()))