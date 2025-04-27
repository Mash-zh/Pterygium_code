import torch
import torchvision.models as models
import torch.nn as nn

# Vgg19 模型
class VGG19Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGG19Model, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的 VGG19 模型
        self.vgg19 = models.vgg19(pretrained=True)
        # 获取输入特征数（VGG19 原始 fc 层输出 1000 类）
        num_features = self.vgg19 .classifier[6].in_features
        # 替换最后一层，全连接层输出 3 类
        self.vgg19.classifier[6] = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.vgg19(x)