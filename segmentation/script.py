import cv2
import torch
from torchvision import transforms

def dice_coefficient_binary(pred, target):
    smooth = 1e-6  # 用于避免除以零
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

class Script():
    def __init__(self):
        pass

    def crop_image_with_overlap(self, image_path, block_size=1024, overlap=128,
                                islabel=False):
        # 读取大图
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        stride = block_size - overlap

        patch_list = []
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # 计算裁剪区域
                x_end = min(x + block_size, w)
                y_end = min(y + block_size, h)

                # 注意处理边界：如果到了边缘，不够512×512，要做padding
                patch = img[y:y_end, x:x_end]

                pad_bottom = block_size - patch.shape[0]
                pad_right = block_size - patch.shape[1]

                if pad_bottom > 0 or pad_right > 0:
                    patch = cv2.copyMakeBorder(
                        patch,
                        0, pad_bottom, 0, pad_right,
                        borderType=cv2.BORDER_CONSTANT,
                        value=[0, 0, 0]  # 黑色padding
                    )
                if islabel:
                    # 将3通道图像转换为灰度图像
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

                    # 将图像转换为二值图（黑色区域为0，其他区域为255）
                    _, patch = cv2.threshold(gray_patch, 1, 255, cv2.THRESH_BINARY)

                patch = transforms.ToTensor()(patch)
                # if islabel:
                    # patch = torch.squeeze(patch).long()
                patch_list.append(patch)

        return patch_list

    def draw_box(self, image_path):
        # 读取图像
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("无法读取图像，请检查路径是否正确")

        # 将图像转换为二值图（黑色区域为0，其他区域为255）
        _, binary_image = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)

        # 找到黑色区域的轮廓
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个与原图相同大小的彩色图像用于绘制矩形框
        image_with_boxes = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # 遍历每个轮廓，绘制最小外接矩形并获取四角坐标
        boxes = []
        for contour in contours:
            # 获取最小外接矩形
            x, y, w, h = cv2.boundingRect(contour)
            # 绘制矩形框
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 获取矩形的四角坐标
            box = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
            boxes.append(box)

        # 显示结果图像
        cv2.imwrite('binary_image.png', binary_image)
        cv2.imwrite('image_with_boxes.png', image_with_boxes)

        return boxes

if __name__ == '__main__':
    script = Script()


