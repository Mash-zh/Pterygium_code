import os

class Config:
    # 路径配置
    DATA_DIR = "./data_task"
    IMAGES_DIR = os.path.join(DATA_DIR, "image")
    MASKS_DIR = os.path.join(DATA_DIR, "mask")
    OUTPUT_DIR = "./outputs"
    LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

    # 训练配置
    SEED = 42
    MAX_EPOCHS = 100
    BATCH_SIZE = 2  # 增加批量大小
    LEARNING_RATE = 1e-4
    VALIDATION_RATIO = 0.2
    EARLY_STOPPING_PATIENCE = 15
    NUM_WORKERS = 1  # 恢复工作进程数

    # 滑动窗口配置
    PATCH_SIZE = (64, 64)  # 极小的patch大小
    OVERLAP = 0  # 无重叠

    # 模型配置
    MODEL = {
        'in_channels': 3,
        'out_channels': 2,
        'channels': (16, 32, 64, 128),  # 恢复通道数
        'strides': (2, 2, 2),
    }

    # 数据增强配置
    AUGMENTATION = {
        'flip_prob': 0.5,
        'rotate_prob': 0.3,
        'rotate_range': 30,
        'noise_prob': 0.2,
        'intensity_scale': 0.15,
        'intensity_shift': 0.1,
    }

    # 训练设置
    USE_AMP = True