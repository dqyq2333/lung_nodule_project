import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split


class MedicalImageDataset(Dataset):
    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 图像已经是 [H, W, C] 格式
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1).float()
        mask = torch.from_numpy(self.masks[idx]).float()

        # 移除多余的维度
        if image.dim() == 4:
            image = image.squeeze(0)
        if mask.dim() == 4:
            mask = mask.squeeze(0)

        return image, mask


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # 编码器
        self.enc1 = self.conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = self.conv_block(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        # 瓶颈层
        self.bottleneck = self.conv_block(256, 512)

        # 解码器
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(512, 256)
        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(256, 128)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(128, 64)
        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(64, 32)

        # 输出层
        self.final = nn.Conv2d(32, 1, kernel_size=1)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def center_crop(self, skip, target):
        _, _, H, W = target.size()
        _, _, h, w = skip.size()

        dh = (h - H) // 2
        dw = (w - W) // 2

        return skip[:, :, dh:dh + H, dw:dw + W]

    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        # 瓶颈
        bottleneck = self.bottleneck(self.pool4(enc4))

        # 解码
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([self.center_crop(enc4, dec4), dec4], dim=1)
        dec4 = self.dec4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([self.center_crop(enc3, dec3), dec3], dim=1)
        dec3 = self.dec3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([self.center_crop(enc2, dec2), dec2], dim=1)
        dec2 = self.dec2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([self.center_crop(enc1, dec1), dec1], dim=1)
        dec1 = self.dec1(dec1)

        return torch.sigmoid(self.final(dec1))


class ModelTrainer:
    def __init__(self,
                 model_path="C:/Users/lenovo/.lmstudio/models/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 只有在需要生成报告时才加载LLM
        self.llm = None
        self.model_path = model_path

    def load_llm(self):
        """按需加载LLM模型"""
        if self.llm is None:
            from llama_cpp import Llama
            self.llm = Llama(model_path=self.model_path, n_ctx=2048)

    def train_unet(self, images, masks, epochs=10, batch_size=2):
        print("输入图像形状:", images.shape)
        print("输入掩码形状:", masks.shape)

        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

        train_dataset = MedicalImageDataset(X_train, y_train)
        val_dataset = MedicalImageDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = UNet().to(self.device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

        best_val_loss = float('inf')

        for epoch in range(epochs):
            model.train()
            train_loss = 0.0

            for images, masks in train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            # 验证
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item() * images.size(0)

            train_loss = train_loss / len(train_loader.dataset)
            val_loss = val_loss / len(val_loader.dataset)

            # 更新学习率
            scheduler.step(val_loss)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "models/best_unet_model.pth")
                print(f"保存最佳模型，验证损失: {val_loss:.4f}")

            torch.cuda.empty_cache()

            print(
                f"Epoch {epoch + 1}/{epochs} - 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 训练结束后清理显存
        del optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

        model.device = self.device
        return model

    def generate_report(self, findings):
        """使用LLM生成报告"""
        if not hasattr(self, 'llm') or self.llm is None:
            self.load_llm()

        prompt = f"作为一位经验丰富的放射科医生，请根据以下影像学发现撰写一份专业的医学报告:\n\n{findings}\n\n报告:"
        response = self.llm(prompt, max_tokens=512)
        return response['choices'][0]['text']