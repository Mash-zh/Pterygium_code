import torch
from vit_pytorch import ViT
import torchvision.models as models
import torch.nn as nn

# Vgg16 模型
class VGG16Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.num_classes = num_classes
        # 加载预训练的 VGG16 模型
        self.vgg16 = models.vgg16(pretrained=True)
        # 获取输入特征数（VGG16 原始 fc 层输出 1000 类）
        num_features = self.vgg16 .classifier[6].in_features
        # 替换最后一层，全连接层输出 3 类
        self.vgg16.classifier[6] = nn.Linear(num_features, self.num_classes)

    def forward(self, x):
        return self.vgg16(x)

# Vision Transformer 模型
class VisionTransformerModel(torch.nn.Module):
    def __init__(self, num_classes):
        super(VisionTransformerModel, self).__init__()
        self.vit = ViT(
            image_size=224,
            patch_size=32,
            num_classes=num_classes,
            dim=1024,
            depth=6,
            heads=16,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x):
        return self.vit(x)


# 结合 EfficientNetB7 和 Vision Transformer 的模型
class VGG16_ViT_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(VGG16_ViT_Model, self).__init__()
        self.vgg16 = VGG16Model(num_classes)
        self.vit = VisionTransformerModel(num_classes)
        self.fc = torch.nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        vgg16_output = self.vgg16(x)
        vit_output = self.vit(x)
        combined_output = torch.cat((vgg16_output, vit_output), dim=1)
        final_output = self.fc(combined_output)
        return final_output

if __name__ == '__main__':
    model = VGG16_ViT_Model(num_classes=3)
    print(model)