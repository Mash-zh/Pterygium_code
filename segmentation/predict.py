import os
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
from typing import Tuple, List
from matplotlib import font_manager

# 设置中文字体
try:
    # 尝试使用微软雅黑
    font_path = 'C:/Windows/Fonts/msyh.ttc'  # 微软雅黑
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/simsun.ttc'  # 尝试宋体
    if not os.path.exists(font_path):
        font_path = 'C:/Windows/Fonts/simhei.ttf'  # 尝试黑体
    
    if os.path.exists(font_path):
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
except Exception as e:
    print(f"警告：设置中文字体失败，将使用默认字体。错误信息：{e}")

def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """
    加载训练好的模型
    
    Args:
        model_path: 模型权重文件路径
        device: 运行设备（CPU/GPU）
    
    Returns:
        加载好权重的模型
    """
    # 初始化模型
    model = smp.UnetPlusPlus(
        encoder_name="mobilenet_v2",
        encoder_weights=None,  # 推理时不需要预训练权重
        in_channels=Config.MODEL['in_channels'],
        classes=Config.MODEL['out_channels'],
        activation=None,
    ).to(device)
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"模型加载成功，最佳Dice分数: {checkpoint['dice_score']:.4f}")
    return model

def preprocess_image(image_path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    预处理输入图像
    
    Args:
        image_path: 图像文件路径
    
    Returns:
        预处理后的图像和原始图像尺寸
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # BGR转RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 保存原始尺寸
    original_size = image.shape[:2]
    
    # 归一化
    image = image.astype(np.float32) / 255.0
    
    return image, original_size

def extract_patches(image: np.ndarray, patch_size: Tuple[int, int], overlap: int) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    将图像分割成重叠的patches
    
    Args:
        image: 输入图像
        patch_size: patch的大小
        overlap: 重叠的像素数
    
    Returns:
        patches列表和对应的坐标信息
    """
    patches = []
    coords = []
    h, w = image.shape[:2]
    stride = patch_size[0] - overlap

    for y in range(0, h-overlap, stride):
        for x in range(0, w-overlap, stride):
            end_y = min(y + patch_size[0], h)
            end_x = min(x + patch_size[1], w)
            start_y = max(0, end_y - patch_size[0])
            start_x = max(0, end_x - patch_size[1])

            patch = image[start_y:end_y, start_x:end_x]
            
            # 如果patch大小不足，进行padding
            if patch.shape[0] != patch_size[0] or patch.shape[1] != patch_size[1]:
                temp_patch = np.zeros((patch_size[0], patch_size[1], image.shape[2]), dtype=image.dtype)
                temp_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = temp_patch

            patches.append(patch)
            coords.append((start_x, start_y, end_x, end_y))

    return patches, coords

def merge_predictions(predictions: List[np.ndarray], coords: List[Tuple[int, int, int, int]], original_size: Tuple[int, int]) -> np.ndarray:
    """
    合并patches的预测结果
    
    Args:
        predictions: 每个patch的预测结果
        coords: 每个patch的坐标信息
        original_size: 原始图像尺寸
    
    Returns:
        合并后的完整预测图
    """
    h, w = original_size
    prediction_map = np.zeros((h, w), dtype=np.float32)
    weight_map = np.zeros((h, w), dtype=np.float32)
    
    for pred, (start_x, start_y, end_x, end_y) in zip(predictions, coords):
        # 创建高斯权重
        patch_h, patch_w = end_y - start_y, end_x - start_x
        y, x = np.ogrid[:patch_h, :patch_w]
        y = y / patch_h - 0.5
        x = x / patch_w - 0.5
        weight = np.exp(-(x*x + y*y) / 0.08)
        
        prediction_map[start_y:end_y, start_x:end_x] += pred * weight
        weight_map[start_y:end_y, start_x:end_x] += weight
    
    # 避免除零错误
    weight_map[weight_map == 0] = 1
    prediction_map = prediction_map / weight_map
    
    return prediction_map

def predict_single_image(model: torch.nn.Module, image_path: str, device: torch.device, output_path: str = None) -> np.ndarray:
    """
    对单张图像进行分割预测
    
    Args:
        model: 加载好的模型
        image_path: 输入图像路径
        device: 运行设备
        output_path: 输出图像保存路径（可选）
    
    Returns:
        分割预测结果
    """
    # 预处理图像
    image, original_size = preprocess_image(image_path)
    
    # 将图像调整为训练尺寸
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    resized_image = F.interpolate(image_tensor, size=(512, 512), mode='bilinear', align_corners=True)
    resized_image = resized_image.to(device)
    
    # 进行预测
    with torch.no_grad():
        with autocast(device_type='cuda', enabled=Config.USE_AMP):
            output = model(resized_image)
            output = torch.softmax(output, dim=1)
            
            # 将预测结果调整回原始尺寸
            output = F.interpolate(output, size=original_size, mode='bilinear', align_corners=True)
            pred = (output[:, 1] > 0.5).float()
    
    # 转回numpy
    final_prediction = pred.squeeze().cpu().numpy()
    
    # 保存结果
    if output_path:
        try:
            # 创建可视化结果
            plt.figure(figsize=(15, 5))
            
            # 显示原始图像
            plt.subplot(131)
            plt.imshow(image)
            plt.title('原始图像', fontproperties=font_prop if 'font_prop' in locals() else None)
            plt.axis('off')
            
            # 显示预测掩码
            plt.subplot(132)
            plt.imshow(final_prediction, cmap='gray')
            plt.title('预测掩码', fontproperties=font_prop if 'font_prop' in locals() else None)
            plt.axis('off')
            
            # 显示叠加结果
            plt.subplot(133)
            overlay = image.copy()
            overlay[final_prediction > 0.5] = [1, 0, 0]  # 红色标记分割区域
            plt.imshow(overlay)
            plt.title('叠加结果', fontproperties=font_prop if 'font_prop' in locals() else None)
            plt.axis('off')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 同时保存二值掩码
            mask_path = output_path.replace('.png', '_mask.png')
            cv2.imwrite(mask_path, (final_prediction > 0.5).astype(np.uint8) * 255)
        except Exception as e:
            print(f"警告：保存可视化结果时出错：{e}")
            # 确保图像被关闭
            plt.close()
    
    return final_prediction

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model_path = "outputs/best_model_dice_0.9001_acc_0.9001.pth"  # 请替换为实际的模型路径
    model = load_model(model_path, device)
    
    # 设置输入输出路径
    image_path = "data_task/image/0091.png"  # 请替换为实际的图像路径
    output_dir = "predictions"
    os.makedirs(output_dir, exist_ok=True)
    
    # 进行预测
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    prediction = predict_single_image(model, image_path, device, output_path)
    print(f"预测完成，结果保存在: {output_path}")

if __name__ == '__main__':
    main()