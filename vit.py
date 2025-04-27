import torch
import torch.nn as nn
from vit_pytorch import ViT

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

if __name__ == '__main__':

    # 示例：创建一个支持3分类的 Vision Transformer 模型
    model = VisionTransformerModel(num_classes=3)
    print(model)