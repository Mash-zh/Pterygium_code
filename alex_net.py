import torch.nn as nn
import torchvision.models as models
import torch

class AlexModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(AlexModel, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的 Alex 模型
        self.alexnet = models.alexnet(pretrained=True)

        # 获取输入特征数（alex 原始 fc 层输出 1000 类）
        num_features = self.alexnet .classifier[6].in_features
        # 替换最后一层，全连接层输出 3 类
        self.alexnet.classifier[6] = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.alexnet(x)

if __name__ == '__main__':
    alex = AlexModel(num_classes=3)
    print(alex)