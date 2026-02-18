import os
import cv2
import numpy as np
from skimage import exposure
import pandas as pd


class DataPreprocessor:
    def __init__(self, data_path, target_size=(256, 256)):
        self.data_path = data_path
        self.image_dir = os.path.join(data_path, 'CXR_png')
        self.mask_dir = os.path.join(data_path, 'ManualMask')
        self.clinical_data = os.path.join(data_path, 'ClinicalReadings')
        self.target_size = (512, 512)  # 统一的目标尺寸

    def load_data(self):
        """加载图像和对应的掩码"""
        images = []
        masks = []

        for filename in os.listdir(self.image_dir):
            if filename.endswith('.png'):
                # 加载图像
                img_path = os.path.join(self.image_dir, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)

                # 加载掩码
                mask_left_path = os.path.join(self.mask_dir, 'leftMask', filename)
                mask_right_path = os.path.join(self.mask_dir, 'rightMask', filename)

                if os.path.exists(mask_left_path) and os.path.exists(mask_right_path):
                    mask_left = cv2.imread(mask_left_path, cv2.IMREAD_GRAYSCALE)
                    mask_right = cv2.imread(mask_right_path, cv2.IMREAD_GRAYSCALE)
                    mask = cv2.bitwise_or(mask_left, mask_right)
                    mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
                    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

                    images.append(img)
                    masks.append(mask)

        # 转换为NumPy数组并确保是4D [N, C, H, W]
        images = np.array(images, dtype=np.float32) / 255.0
        masks = np.array(masks, dtype=np.float32) / 255.0

        # 添加通道维度 (从 [N,H,W] 变为 [N,1,H,W])
        images = np.expand_dims(images, axis=1)
        masks = np.expand_dims(masks, axis=1)

        return images, masks

    def preprocess_images(self, images):
        """图像预处理：降噪、增强等"""
        processed_images = []

        for img in images:
            # 去除通道维度（如果是单通道）
            if len(img.shape) > 2:
                img = img.squeeze()

            # 直方图均衡化
            img_eq = exposure.equalize_hist(img)

            # 高斯模糊降噪
            img_blur = cv2.GaussianBlur(img_eq, (5, 5), 0)

            # 归一化
            img_normalized = (img_blur - np.min(img_blur)) / (np.max(img_blur) - np.min(img_blur))

            # 重新添加通道维度
            img_normalized = np.expand_dims(img_normalized, axis=-1)

            processed_images.append(img_normalized)

        return np.array(processed_images, dtype=np.float32)

    def load_clinical_data(self):
        """加载临床数据"""
        clinical_files = [f for f in os.listdir(self.clinical_data) if f.endswith('.csv')]
        dfs = []

        for file in clinical_files:
            df = pd.read_csv(os.path.join(self.clinical_data, file))
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)