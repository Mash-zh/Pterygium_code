import torch
from efficientnet_pytorch import EfficientNet
from vit_pytorch import ViT

# EfficientNetB7 模型
class EfficientNetB7Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7Model, self).__init__()
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)

    def forward(self, x):
        return self.efficientnet(x)

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
class EfB7_ViT_Model(torch.nn.Module):
    def __init__(self, num_classes):
        super(EfB7_ViT_Model, self).__init__()
        self.efficientnet = EfficientNetB7Model(num_classes)
        self.vit = VisionTransformerModel(num_classes)
        self.fc = torch.nn.Linear(num_classes * 2, num_classes)

    def forward(self, x):
        efficientnet_output = self.efficientnet(x)
        vit_output = self.vit(x)
        combined_output = torch.cat((efficientnet_output, vit_output), dim=1)
        final_output = self.fc(combined_output)
        return final_output

if __name__ == '__main__':
    model = EfB7_ViT_Model(num_classes=3)
    print(model)