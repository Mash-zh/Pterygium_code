import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import random
from config import Config
import cv2
from torch.utils.data import Dataset
from glob import glob
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torch.utils.data import RandomSampler

gen = torch.Generator()
gen.manual_seed(Config.SEED)  # 固定随机种子

from monai.transforms import (
    LoadImaged,            # 载入字典中指定 keys 的文件
    EnsureChannelFirstd,   # 确保通道维在第一维
    ScaleIntensityd,       # 对 image 做归一化
    RandFlipd,             # 随机翻转（同步 image & mask）
    RandRotated,           # 随机旋转（同步 image & mask）
    RandGaussianNoised,    # 随机添加高斯噪声
    RandScaleIntensityd,   # 随机缩放图像强度
    RandShiftIntensityd,   # 随机偏移图像强度
    RandRotate90d,         # 随机旋转90度
    Resized,               # 调整图像大小
    ToTensord,             # 转为 Tensor
    Compose,
    AsDiscreted,           # 将连续值转换为离散值
    MapTransform,
    RandAffined,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandHistogramShiftd
)

from monai.data import CacheDataset, list_data_collate
from monai.networks.nets import UNet
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_model = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_transforms():
    train_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        Resized(
            keys=["image", "mask"],
            spatial_size=(512, 512),  # 训练时使用较小的尺寸
            mode=("bilinear", "nearest")
        ),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ScaleIntensityd(keys=["mask"], minv=0.0, maxv=1.0),
        AsDiscreted(keys=["mask"], threshold=0.5),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["mask"]),
        RandFlipd(
            keys=["image", "mask"],
            prob=Config.AUGMENTATION['flip_prob'],
            spatial_axis=[0, 1]
        ),
        RandRotated(
            keys=["image", "mask"],
            range_x=Config.AUGMENTATION['rotate_range'],
            prob=Config.AUGMENTATION['rotate_prob'],
            mode=("bilinear", "nearest")
        ),
        RandAffined(
            keys=["image", "mask"],
            prob=0.5,
            rotate_range=(0, 0, np.pi/15),
            scale_range=(0.1, 0.1),
            mode=("bilinear", "nearest"),
        ),
        RandScaleIntensityd(
            keys=["image"],
            factors=Config.AUGMENTATION['intensity_scale'],
            prob=0.5,
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=Config.AUGMENTATION['intensity_shift'],
            prob=0.5,
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=Config.AUGMENTATION['noise_prob'],
            mean=0.0,
            std=0.1,
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=0.3,
            gamma=(0.7, 1.3),
        ),
        RandGaussianSmoothd(
            keys=["image"],
            prob=0.2,
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
        ),
        RandHistogramShiftd(
            keys=["image"],
            prob=0.3,
            num_control_points=10,
        ),
        ToTensord(keys=["image", "mask"]),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "mask"]),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        ScaleIntensityd(keys=["mask"], minv=0.0, maxv=1.0),
        AsDiscreted(keys=["mask"], threshold=0.5),
        ConvertToMultiChannelBasedOnBratsClassesd(keys=["mask"]),
        ToTensord(keys=["image", "mask"]),
    ])

    return train_transforms, val_transforms

def prepare_data():
    data_dicts = [
        {
            "image": os.path.join(Config.IMAGES_DIR, f),
            "mask": os.path.join(Config.MASKS_DIR, f[:-4] + '_mask.png')
        }
        for f in os.listdir(Config.IMAGES_DIR)
        if os.path.exists(os.path.join(Config.MASKS_DIR, f[:-4] + '_mask.png'))
    ]
    
    random.shuffle(data_dicts)
    val_size = int(len(data_dicts) * Config.VALIDATION_RATIO)
    
    return data_dicts[val_size:], data_dicts[:val_size]

def get_dataloaders(train_files, val_files, train_transforms, val_transforms):
    train_ds = CacheDataset(
        data=train_files,
        transform=train_transforms,
        cache_rate=0.0
    )

    train_fullsize_ds = CacheDataset(
        data=train_files,
        transform=val_transforms,
        cache_rate=0.0
    )
    val_ds = CacheDataset(
        data=val_files,
        transform=val_transforms,
        cache_rate=0.0
    )

    # 共享同一个随机种子
    gen = torch.Generator()
    gen.manual_seed(Config.SEED)
    # 用同一个Generator构造两个RandomSampler
    sampler1 = RandomSampler(train_ds, replacement=False, generator=gen)
    # 复用同样的种子构造第二个 sampler
    gen2 = torch.Generator()
    gen2.manual_seed(Config.SEED)
    sampler2 = RandomSampler(train_fullsize_ds, replacement=False, generator=gen2)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler1,
        num_workers=Config.NUM_WORKERS,
        collate_fn=list_data_collate
    )

    train_fullsize_loader = DataLoader(
        train_fullsize_ds,
        batch_size=Config.BATCH_SIZE,
        sampler=sampler2,
        num_workers=Config.NUM_WORKERS,
        collate_fn=list_data_collate
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=list_data_collate
    )
    
    return train_loader, train_fullsize_loader, val_loader

def setup_logger():
    # 创建logger
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建文件处理器
    log_filename = f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(os.path.join(Config.OUTPUT_DIR, log_filename))
    file_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # 添加处理器到logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def train_epoch(model, train_loader, train_fullsize_loader, criterion, optimizer, scaler, epoch, logger):
    model.train()
    epoch_loss = 0
    pbar = tqdm(zip(train_loader, train_fullsize_loader),
                total=len(train_loader),
                desc=f'Epoch {epoch + 1}')
    for batch_small, batch_full in pbar:
        images = batch_small["image"].cuda()
        masks = batch_small["mask"].cuda()
        full_masks = batch_full["mask"].cuda()
        # 获取原始图像尺寸
        original_size = batch_full["mask"].shape[2:]
        optimizer.zero_grad(set_to_none=True)  # 更高效的梯度清零

        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            outputs = model(images)
            # 将输出调整回原始尺寸
            full_outputs = F.interpolate(outputs, size=original_size, mode='bilinear', align_corners=True)
            loss_resize = criterion(outputs, masks)
            loss_full = criterion(full_outputs, full_masks)
            loss = loss_resize*1 + loss_full*0
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # 立即清理不需要的张量
        del outputs
        torch.cuda.empty_cache()

        current_loss = loss.item()
        epoch_loss += current_loss

        gpu_memory = torch.cuda.memory_allocated(0)/1024**2
        gpu_memory_cached = torch.cuda.memory_reserved(0)/1024**2
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'GPU': f'{gpu_memory:.0f}MB/{gpu_memory_cached:.0f}MB'
        })

        if pbar.n % 100 == 0:
            logger.info(f'Epoch {epoch + 1}, Batch {pbar.n}/{len(train_loader)}, '
                      f'Loss: {current_loss:.4f}, GPU使用: {gpu_memory:.0f}MB/{gpu_memory_cached:.0f}MB')
    
    return epoch_loss / len(train_loader)

# def calculate_metrics(pred, target):
#     """
#     计算多个评估指标
#     """
#     with torch.no_grad():
#         # 获取预测的类别
#         pred_class = (pred[:, 1] > 0.5).float()
#         target_class = target[:, 1]
#
#         # 计算TP, FP, TN, FN
#         tp = torch.sum((pred_class == 1) & (target_class == 1)).float()
#         fp = torch.sum((pred_class == 1) & (target_class == 0)).float()
#         tn = torch.sum((pred_class == 0) & (target_class == 0)).float()
#         fn = torch.sum((pred_class == 0) & (target_class == 1)).float()
#
#         # 计算各种指标
#         accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)
#         precision = tp / (tp + fp + 1e-7)
#         recall = tp / (tp + fn + 1e-7)
#
#         # 直接计算Dice系数
#         dice_score = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
#
#         # IoU (Intersection over Union)
#         iou = tp / (tp + fp + fn + 1e-7)
#
#         return {
#             'accuracy': accuracy.item(),
#             'precision': precision.item(),
#             'recall': recall.item(),
#             'f1_score': dice_score.item(),  # f1_score 等同于 dice_score
#             'iou': iou.item()
#         }

def validate(model, val_loader, criterion, logger):
    model.eval()
    val_loss = 0
    dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    resize_dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
    num_batches = 0
    
    gpu_memory = torch.cuda.memory_allocated(0)/1024**2
    gpu_memory_cached = torch.cuda.memory_reserved(0)/1024**2
    logger.info(f'验证开始时GPU使用: {gpu_memory:.0f}MB/{gpu_memory_cached:.0f}MB')
    
    with torch.no_grad():
        for batch_data in val_loader:
            # 获取原始图像尺寸
            original_size = batch_data["image"].shape[2:]
            
            # 调整图像大小用于推理
            resized_image = F.interpolate(batch_data["image"], size=(512, 512), mode='bilinear', align_corners=True)
            resized_image = resized_image.cuda()
            masks = batch_data["mask"].cuda()
            resized_masks = F.interpolate(masks, size=(512, 512), mode='bilinear', align_corners=True)
            resized_masks = resized_masks.cuda()

            with autocast(device_type='cuda', enabled=Config.USE_AMP):
                # 在较小尺寸上进行推理
                outputs = model(resized_image)
                resized_outputs = outputs
                # 将输出调整回原始尺寸
                outputs = F.interpolate(outputs, size=original_size, mode='bilinear', align_corners=True)
                
                loss = criterion(outputs, masks)
                val_loss += loss.item()
            
            # 应用softmax获取概率
            outputs = torch.softmax(outputs, dim=1)
            resized_outputs = torch.softmax(resized_outputs, dim=1)
            # 计算Dice分数
            pred = (outputs[:, 1:] > 0.5).float()
            resized_pred = (resized_outputs[:, 1:] > 0.5).float()
            dice_metric(y_pred=pred, y=masks[:, 1:])
            resize_dice_metric(y_pred=resized_pred, y=resized_masks[:, 1:])
            
            num_batches += 1
    
    # 计算平均值
    val_loss = val_loss / len(val_loader)
    dice_score = dice_metric.aggregate().item()
    resized_dice_score = resize_dice_metric.aggregate().item()
    
    # 重置度量计算器
    dice_metric.reset()
    resize_dice_metric.reset()
    
    # 记录指标
    logger.info(f'Validation - Loss: {val_loss:.4f}, Dice: {dice_score:.4f}, resized_Dice:{resized_dice_score:.4f}')
    
    return val_loss, dice_score, resized_dice_score, {'accuracy': dice_score}  # 为了保持接口一致，返回一个字典

class PterygiumDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transforms=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def extract_patches(self, image, mask):
        patch_size = Config.PATCH_SIZE[0]
        overlap = Config.OVERLAP

        patches = []
        mask_patches = []
        coords = []  # 记录每个patch的坐标
        h, w = image.shape[:2]
        stride = patch_size - overlap

        for y in range(0, h-overlap, stride):
            for x in range(0, w-overlap, stride):
                end_y = min(y + patch_size, h)
                end_x = min(x + patch_size, w)
                start_y = max(0, end_y - patch_size)
                start_x = max(0, end_x - patch_size)

                patch = image[start_y:end_y, start_x:end_x]
                mask_patch = mask[start_y:end_y, start_x:end_x]
                
                # 如果patch大小不足，进行padding
                if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                    temp_patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
                    temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = temp_patch

                    temp_mask = np.zeros((patch_size, patch_size), dtype=mask.dtype)
                    temp_mask[:mask_patch.shape[0], :mask_patch.shape[1]] = mask_patch
                    mask_patch = temp_mask

                patches.append(patch)
                mask_patches.append(mask_patch)
                coords.append((start_x, start_y, end_x, end_y))

        return patches, mask_patches, coords

    def __getitem__(self, idx):
        # 读取原始图像和掩码
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        data = {
            "image": image_path,
            "mask": mask_path
        }
        
        if self.transforms:
            data = self.transforms(data)
            
        return data

class ConvertToMultiChannelBasedOnBratsClassesd(MapTransform):
    """
    将单通道掩码转换为多通道格式
    """
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            if mask.shape[0] == 1:
                # 创建背景通道和前景通道
                background = (mask == 0).float()
                foreground = (mask == 1).float()
                d[key] = torch.cat([background, foreground], dim=0)
        return d

def save_prediction_results(epoch, val_dice, val_accuracy, model, val_loader, device, logger):
    """
    随机选择一个验证集图像进行推理并保存结果，同时保存原始大小和缩放后的结果
    """
    # 创建保存目录
    save_dir = os.path.join(Config.OUTPUT_DIR, f'epoch_{epoch}_dice_{val_dice:.4f}_acc_{val_accuracy:.4f}')
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择一个验证集样本
    val_data_list = list(val_loader)
    val_data = random.choice(val_data_list)
    
    # 获取原始图像尺寸
    original_size = val_data["image"].shape[2:]
    
    # 设置可视化尺寸（用于显示）
    vis_size = (1024, 768)  # 更适合显示的尺寸
    
    # 调整图像大小用于推理
    resized_image = F.interpolate(val_data["image"], size=(512, 512), mode='bilinear', align_corners=True)
    resized_image = resized_image.to(device)
    mask = val_data["mask"].to(device)
    
    # 进行推理
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            output = model(resized_image)
            # 将输出调整回原始尺寸
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
            output = torch.softmax(output, dim=1)
            # 使用阈值0.5获取二值预测
            pred = (output[:, 1] > 0.5).float()
    
    # 转换为numpy数组
    image_np = val_data["image"][0].cpu().numpy().transpose(1, 2, 0)  # [H, W, C]
    mask_np = mask[0, 1].cpu().numpy()  # 取第二个通道（前景）
    pred_np = pred[0].cpu().numpy()
    
    # 将预测和掩码转换为严格的二值图像（0或255）
    mask_binary = np.where(mask_np > 0.5, 255, 0).astype(np.uint8)
    pred_binary = np.where(pred_np > 0.5, 255, 0).astype(np.uint8)
    
    # 归一化原始图像到0-255范围
    image_np = (image_np * 255).astype(np.uint8)
    
    # 保存原始大小的图像和掩码
    cv2.imwrite(os.path.join(save_dir, 'original_full.png'), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dir, 'mask_full.png'), mask_binary)
    cv2.imwrite(os.path.join(save_dir, 'prediction_full.png'), pred_binary)
    
    # 调整图像大小用于可视化
    image_vis = cv2.resize(image_np, vis_size, interpolation=cv2.INTER_AREA)
    mask_vis = cv2.resize(mask_binary, vis_size, interpolation=cv2.INTER_NEAREST)
    pred_vis = cv2.resize(pred_binary, vis_size, interpolation=cv2.INTER_NEAREST)
    
    # 创建可视化图像（使用调整后的尺寸）
    plt.figure(figsize=(15, 5))
    
    # 原图
    plt.subplot(131)
    plt.imshow(image_vis)
    plt.title('Original Image')
    plt.axis('off')
    
    # 真实掩码
    plt.subplot(132)
    plt.imshow(mask_vis, cmap='binary')
    plt.title('Ground Truth Mask')
    plt.axis('off')
    
    # 预测掩码
    plt.subplot(133)
    plt.imshow(pred_vis, cmap='binary')
    plt.title('Predicted Mask')
    plt.axis('off')
    
    # 保存对比图
    plt.savefig(os.path.join(save_dir, 'comparison.png'), bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 为了更好地可视化，也保存带有轮廓的原图（使用调整后的尺寸）
    overlay = image_vis.copy()
    # 绘制真实掩码的轮廓（绿色）
    contours, _ = cv2.findContours(mask_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    # 绘制预测掩码的轮廓（红色）
    contours, _ = cv2.findContours(pred_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 0, 0), 2)
    # 保存带有轮廓的图像
    cv2.imwrite(os.path.join(save_dir, 'overlay.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    # 创建一个文本文件记录图像信息
    with open(os.path.join(save_dir, 'image_info.txt'), 'w') as f:
        f.write(f'原始图像尺寸: {original_size}\n')
        f.write(f'可视化图像尺寸: {vis_size}\n')
        f.write(f'Dice分数: {val_dice:.4f}\n')
        f.write(f'准确率: {val_accuracy:.4f}\n')
    
    logger.info(f'保存预测结果到: {save_dir}')
    logger.info(f'原始尺寸: {original_size}, 可视化尺寸: {vis_size}')

def main():
    # try:
    # 设置CUDA内存分配器配置
    torch.cuda.set_per_process_memory_fraction(0.7)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 清理GPU缓存
    torch.cuda.empty_cache()

    # 初始化设置
    set_seed(Config.SEED)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    # 检查数据目录是否存在
    if not os.path.exists(Config.IMAGES_DIR):
        raise FileNotFoundError(f"图像目录不存在: {Config.IMAGES_DIR}")
    if not os.path.exists(Config.MASKS_DIR):
        raise FileNotFoundError(f"掩码目录不存在: {Config.MASKS_DIR}")

    # 检查数据目录中是否有文件
    if len(os.listdir(Config.IMAGES_DIR)) == 0:
        raise ValueError(f"图像目录为空: {Config.IMAGES_DIR}")
    if len(os.listdir(Config.MASKS_DIR)) == 0:
        raise ValueError(f"掩码目录为空: {Config.MASKS_DIR}")

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger()

    # 显示训练设备信息
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU型号: {torch.cuda.get_device_name(0)}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
        logger.info(f"当前GPU显存使用: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
        logger.info(f"当前GPU显存缓存: {torch.cuda.memory_reserved(0)/1024**2:.2f} MB")

    writer = SummaryWriter(Config.LOG_DIR)

    # 准备数据
    train_transforms, val_transforms = get_transforms()
    train_files, val_files = prepare_data()
    train_loader, train_fullsize_loader, val_loader = get_dataloaders(
        train_files, val_files, train_transforms, val_transforms
    )

    # 初始化模型
    model = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_weights="imagenet",
        in_channels=Config.MODEL['in_channels'],
        classes=Config.MODEL['out_channels'],
        activation=None,
    ).to(device)

    # 使用组合损失函数
    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = smp.losses.DiceLoss(mode='binary')
    focal_loss = smp.losses.FocalLoss(mode='binary')

    def criterion(pred, target):
        return 0.4 * bce_loss(pred, target) + \
               0.4 * dice_loss(pred, target) + \
               0.2 * focal_loss(pred, target)

    # 使用AdamW优化器和余弦退火学习率调度
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )

    scaler = GradScaler()

    logger.info(f"训练集大小: {len(train_loader.dataset)}, 全尺寸训练集大小: {len(train_fullsize_loader.dataset)}, 验证集大小: {len(val_loader.dataset)}")
    logger.info(f"开始训练...")

    # 训练循环
    best_dice = 0
    best_model_path = None
    patience_counter = 0

    for epoch in range(Config.MAX_EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{Config.MAX_EPOCHS}")
        train_loss = train_epoch(model, train_loader, train_fullsize_loader, criterion, optimizer, scaler, epoch, logger)
        val_loss, val_dice, resized_val_dice, val_metrics = validate(model, val_loader, criterion, logger)

        # 记录指标
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Dice/val', val_dice, epoch)
        writer.add_scalar('resized_Dice/val', resized_val_dice, epoch)
        writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)

        logger.info(f'Epoch {epoch + 1} Summary:')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        logger.info(f'Val Dice: {val_dice:.4f}, Val Accuracy: {val_metrics["accuracy"]:.4f}')
        logger.info(f'Val resized_Dice: {resized_val_dice:.4f}')

        # 保存预测结果
        save_prediction_results(epoch + 1, val_dice, val_metrics['accuracy'], model, val_loader, device, logger)

        # 只保存最佳模型
        if val_dice > best_dice:
            best_dice = val_dice
            # 删除之前的最佳模型
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            # 保存新的最佳模型
            best_model_path = os.path.join(Config.OUTPUT_DIR, f'best_model_dice_{val_dice:.4f}_acc_{val_metrics["accuracy"]:.4f}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'dice_score': val_dice,
                'accuracy': val_metrics['accuracy']
            }, best_model_path)
            logger.info(f'保存新的最佳模型 (Dice: {val_dice:.4f}, Accuracy: {val_metrics["accuracy"]:.4f})')
            patience_counter = 0
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            logger.info(f'Early stopping triggered after {epoch + 1} epochs')
            break

        # 更新学习率
        scheduler.step()

    writer.close()
    logger.info(f'训练完成!')
    logger.info(f'最佳验证指标 - Dice: {best_dice:.4f}')
        
    # except Exception as e:
    #     logger.error(f"训练过程中发生错误: {e}")

if __name__ == '__main__':
    main()