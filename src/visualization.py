import gradio as gr
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import os


class Visualizer:
    def __init__(self, model, trainer):
        self.model = model
        self.trainer = trainer
        self.device = next(model.parameters()).device
        self.results_dir = "predictions"
        os.makedirs(self.results_dir, exist_ok=True)

    def visualize_results(self, image, prediction, ground_truth=None):
        """可视化原始图像、预测结果和真实掩码（如果可用）"""
        fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))

        # 原始图像
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("原始图像")
        axes[0].axis('off')

        # 预测结果
        axes[1].imshow(prediction, cmap='jet')
        axes[1].set_title("预测结果")
        axes[1].axis('off')

        # 真实掩码（如果可用）
        if ground_truth is not None:
            axes[2].imshow(ground_truth, cmap='gray')
            axes[2].set_title("真实掩码")
            axes[2].axis('off')

        # 保存图像
        plt.savefig(os.path.join(self.results_dir, 'prediction_result.png'))
        plt.close(fig)

        return fig

    def process_image(self, image):
        try:
            if image is None or image.size == 0:
                raise ValueError("上传的图像为空或无效")

            # 记录原始尺寸
            original_shape = image.shape[:2]

            # 转换为灰度图
            if len(image.shape) == 2:
                image_gray = image
            elif len(image.shape) == 3:
                if image.shape[2] == 3:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 4:
                    image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
                else:
                    raise ValueError(f"不支持的图像通道数: {image.shape}")
            else:
                raise ValueError(f"不支持的图像维度: {image.shape}")

            # 归一化
            image_processed = (image_gray - np.min(image_gray)) / (np.max(image_gray) - np.min(image_gray))

            # 缩放为512x512
            resized_image = cv2.resize(image_processed, (512, 512), interpolation=cv2.INTER_AREA)

            # 转换为模型输入
            image_tensor = torch.FloatTensor(resized_image).unsqueeze(0).unsqueeze(0).to(self.device)

            # 预测
            with torch.no_grad():
                prediction = self.model(image_tensor).squeeze().cpu().numpy()

            # 将预测结果缩回原始尺寸
            prediction = cv2.resize(prediction, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

            # 应用阈值
            prediction_binary = (prediction > 0.5).astype(np.uint8) * 255

            # 清理显存
            del image_tensor
            torch.cuda.empty_cache()

            # 生成报告
            findings = "胸部X光显示肺部区域有异常密度影。"
            report = self.trainer.generate_report(findings) if hasattr(self.trainer, 'generate_report') else "报告生成功能不可用"

            # 可视化
            fig = self.visualize_results(image_gray, prediction_binary)

            return fig, report
        except Exception as e:
            print(f"图像处理错误: {e}")
            torch.cuda.empty_cache()
            return None, f"错误: {str(e)}"

    def create_gradio_interface(self):
        def safe_process_image(image):
            try:
                if image is None:
                    raise ValueError("未接收到图像")
                return self.process_image(image)
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, f"处理错误: {str(e)}"

        return gr.Interface(
            fn=safe_process_image,
            inputs=gr.Image(label="上传胸部X光片", type="numpy"),
            outputs=[
                gr.Plot(label="分割结果"),
                gr.Textbox(label="医学报告")
            ],
            title="医学图像分析系统",
            description="上传胸部X光片进行肺结节检测和分析"
        )