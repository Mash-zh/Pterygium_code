import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import shutil
from tqdm import tqdm
from glob import glob
import numpy as np

# 定义类别映射
CLASS_NAMES = {
    0: "正常",
    1: "翼状胬肉建议观察",
    2: "翼状胬肉建议手术"
}

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    预处理输入图像

    Args:
        image_path: 图像文件路径

    Returns:
        预处理后的图像张量
    """
    # 定义图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # 读取并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # 添加batch维度

def predict_single_image(model: nn.Module, image_path: str, device: torch.device) -> int:
    """
    对单张图像进行分类预测

    Args:
        model: 加载好的模型
        image_path: 输入图像路径
        device: 运行设备

    Returns:
        预测的类别（0, 1, 或 2）
    """
    # 预处理图像
    image_tensor = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
    return predicted.item()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 初始化VGG16模型
    model = models.vgg16(pretrained=False)
    # 获取输入特征数（VGG16 原始 fc 层输出 1000 类）
    num_features = model.classifier[6].in_features
    # 替换最后一层，全连接层输出 3 类
    model.classifier[6] = nn.Linear(num_features, 3)
    model = model.to(device)
    
    # 加载模型权重
    model_path = "VGG16_max_acc_0.9555555555555556.pth"
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 设置输入输出路径
    input_dir = "val_img"
    output_dir = "Classification_Results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建类别子文件夹
    for class_id in CLASS_NAMES.keys():
        class_dir = os.path.join(output_dir, str(class_id))
        os.makedirs(class_dir, exist_ok=True)
    
    # 获取所有输入图像
    image_files = glob(os.path.join(input_dir, "*.*"))
    
    # 准备存储结果的列表
    results = []
    
    # 对每张图像进行预测
    for image_path in tqdm(image_files, desc="处理图像"):
        # 获取文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 进行预测
        predicted_class = predict_single_image(model, image_path, device)
        
        # 保存结果
        results.append({
            "样本编号": base_name,
            "预测类别": predicted_class
        })
        
        # 复制图像到对应类别文件夹
        target_path = os.path.join(output_dir, str(predicted_class), os.path.basename(image_path))
        shutil.copy2(image_path, target_path)
        
        print(f"已处理: {image_path} -> 类别 {predicted_class} ({CLASS_NAMES[predicted_class]})")
    
    # 创建DataFrame并保存到Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(output_dir, "Classification_Results.xlsx")
    df.to_excel(excel_path, index=False)
    print(f"\n分类结果已保存到: {excel_path}")
    
    # 打印每个类别的统计信息
    class_counts = df["预测类别"].value_counts()
    print("\n各类别统计信息：")
    for class_id, count in class_counts.items():
        print(f"类别 {class_id} ({CLASS_NAMES[class_id]}): {count}张图像")

if __name__ == '__main__':
    main()