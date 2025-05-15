import torch
import torch.nn as nn
import torchvision.models as models
from EfB7_ViT import EfB7_ViT_Model
from vgg16_ViT import VGG16_ViT_Model
from vit import VisionTransformerModel
from alex_net import AlexModel
from vgg19 import VGG19Model

class customModel(nn.Module):
    def __init__(self, model_name, num_class=3, ispretrained=True):
        super().__init__()
        self.model_name = model_name
        self.num_class = num_class
        self.ispretrained = ispretrained
        self.model = self.__net__()

    def __net__(self):
        if self.model_name == "VGG16":
            # 加载预训练的 VGG16 模型
            model = models.vgg16(pretrained=self.ispretrained)
            # 获取输入特征数（VGG16 原始 fc 层输出 1000 类）
            num_features = model.classifier[6].in_features
            # 替换最后一层，全连接层输出 3 类
            model.classifier[6] = nn.Linear(num_features, self.num_class)
        elif self.model_name == "VGG19":
            model = VGG19Model(num_classes=self.num_class)

        elif self.model_name == "EfB7_ViT":
            model = EfB7_ViT_Model(self.num_class)

        elif self.model_name == "VGG16_ViT":
            model = VGG16_ViT_Model(self.num_class)

        elif self.model_name == "ViT":
            model = VisionTransformerModel(num_classes=self.num_class)

        elif self.model_name == "AlexNet":
            model = AlexModel(num_classes=self.num_class)

        elif self.model_name == "GoogleNet":
            model = models.GoogLeNet(num_classes=self.num_class)

        elif self.model_name == "inception_v3":
            model = models.inception_v3(num_classes = self.num_class)

        elif self.model_name == "resnet18":
            model = models.resnet18(num_classes = self.num_class)

        elif self.model_name == "resnet34":
            model = models.resnet34(num_classes=self.num_class)

        elif self.model_name == "resnet50":
            model = models.resnet50(num_classes=self.num_class)

        elif self.model_name == "resnet101":
            model = models.resnet101(num_classes = self.num_class)

        elif self.model_name == "resnet152":
            model = models.resnet152(num_classes = self.num_class)

        else:
            print("no this model!")

        return model
        # 打印模型，检查修改是否生效
        # print(model)

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = customModel(model_name="AlexNet")
    print(model)