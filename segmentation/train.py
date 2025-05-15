import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from monai.transforms import (
    LoadImaged,            # 载入字典中指定 keys 的文件
    EnsureChannelFirstD,   # 确保通道维在第一维
    ScaleIntensityd,       # 对 image 做归一化
    RandFlipd,             # 随机翻转（同步 image & mask）
    RandRotated,           # 随机旋转（同步 image & mask）
    ToTensord,             # 转为 Tensor
    Compose
)

from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

# ================= Configuration =================
data_dir = "./data_task"                # 数据根目录
images_dir = os.path.join(data_dir, "image")
masks_dir  = os.path.join(data_dir, "mask")
large_image_path = "./data_task/image/0001.png"  # 大图推理路径
output_dir = "./outputs"
os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = 'cpu'

# Training parameters
max_epochs      = 100
train_batch     = 1
val_batch       = 1
learning_rate   = 1e-4
roi_size        = (256, 256)
sw_batch_size   = 1
overlap         = 0.15
use_amp         = True  # 混合精度

# 1. Transforms
train_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),                 # 读入字典里的 image 和 mask
    EnsureChannelFirstD(keys=["image", "mask"]),        # 保证通道维度在第一位
    ScaleIntensityd(keys=["image"]),                    # 仅归一化图像，不对 mask 做强度变换
    RandFlipd(keys=["image", "mask"], prob=0.5, spatial_axis=0),
    RandRotated(keys=["image", "mask"], range_x=15, prob=0.5),
    ToTensord(keys=["image", "mask"]),                  # 转为 Tensor
])

val_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),              # 读取 image/mask 两个键对应的路径
    EnsureChannelFirstD(keys=["image", "mask"]),     # [H,W] → [C,H,W]
    ScaleIntensityd(keys=["image"]),                 # 仅归一化图像
    ToTensord(keys=["image", "mask"]),               # 转为 Tensor, 保留 mask 整数标签
])

infer_transforms = Compose([
    LoadImaged(keys=["image", "mask"]),              # 读取 image/mask 两个键对应的路径
    EnsureChannelFirstD(keys=["image", "mask"]),     # [H,W] → [C,H,W]
    ScaleIntensityd(keys=["image"]),                 # 仅归一化图像
    ToTensord(keys=["image", "mask"]),               # 转为 Tensor, 保留 mask 整数标签
])

# 2. Dataset and DataLoader
data_dicts = []
for f in sorted(os.listdir(images_dir)):
    img_path = os.path.join(images_dir, f)
    mask_name = f[:-4] + '_mask.png'
    msk_path = os.path.join(masks_dir,  mask_name)
    if os.path.exists(msk_path):
        data_dicts.append({"image": img_path, "mask": msk_path})
# split
train_files = data_dicts[:-60]
val_files   = data_dicts[-60:]

train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=0.0)
val_ds   = CacheDataset(data=val_files,   transform=val_transforms,   cache_rate=0.0)

train_loader = DataLoader(train_ds, batch_size=train_batch, shuffle=True,  collate_fn=list_data_collate)
val_loader   = DataLoader(val_ds,   batch_size=val_batch, shuffle=False, collate_fn=list_data_collate)

def main():
    # 3. Model, Loss, Optimizer, Metric
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=2,
        channels=(32, 64, 128, 256, 512),
        strides=(2,2,2,2),
        num_res_units=2,
    ).to(device)

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer     = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dice_metric   = DiceMetric(include_background=True, reduction="mean")

    # 4. Training and Validation loop
    scaler = GradScaler(device=device, enabled=use_amp)
    max_dice = 0
    for epoch in range(1, max_epochs+1):
        torch.cuda.empty_cache()
        # Training
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            images, masks = batch["image"].to(device), batch["mask"].to(device)
            optimizer.zero_grad()
            masks = masks // 255
            masks = masks.long()  # 类型转换
            with autocast(device_type='cuda', enabled=use_amp):
                outputs = model(images)
                loss = loss_function(outputs, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        dice_metric.reset()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                images, masks = batch["image"].to(device), batch["mask"].to(device)
                outputs = model(images)
                outputs = torch.softmax(outputs, 1)
                dice_metric(y_pred=outputs, y=masks)
        val_dice = dice_metric.aggregate().item()

        print(f"Epoch {epoch:03d} - train_loss: {train_loss:.4f}, val_dice: {val_dice:.4f}")
        if max_dice < val_dice:
            max_dice = val_dice

            # 5. Save model
            model_path = os.path.join(output_dir, ("unet_model_" + str(max_dice) + ".pth"))
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at: {model_path}")

    # 6. Inference on large image
    model.eval()
    # preprocess large image
    large_img = infer_transforms(large_image_path).unsqueeze(0).to(device)
    with torch.no_grad():
        if use_amp:
            with autocast():
                pred = sliding_window_inference(
                    inputs=large_img,
                    roi_size=roi_size,
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=overlap,
                    mode='gaussian',
                )
        else:
            pred = sliding_window_inference(
                inputs=large_img,
                roi_size=roi_size,
                sw_batch_size=sw_batch_size,
                predictor=model,
                overlap=overlap,
                mode='constant',
            )
    # postprocess and save mask
    mask = torch.argmax(pred, dim=1)[0].cpu().numpy().astype(np.uint8)
    mask_img = Image.fromarray(mask * 255)
    mask_img.save(os.path.join(output_dir, "large_pred.png"))
    print("Inference result saved.")

if __name__ == '__main__':
    main()