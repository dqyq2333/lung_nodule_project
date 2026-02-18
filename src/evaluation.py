import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import pandas as pd
import seaborn as sns
import os


class Evaluator:
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

    def evaluate_model(self, test_loader):
        """评估模型性能并返回多个指标"""
        self.model.eval()
        all_preds = []
        all_masks = []
        all_probs = []

        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images)
                preds = (outputs > 0.5).float()
                probs = outputs

                all_preds.append(preds.cpu().numpy())
                all_masks.append(masks.cpu().numpy())
                all_probs.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_masks = np.concatenate(all_masks)
        all_probs = np.concatenate(all_probs)

        # 展平数组以计算指标
        preds_flat = all_preds.flatten()
        masks_flat = all_masks.flatten()
        probs_flat = all_probs.flatten()

        # 计算指标
        accuracy = accuracy_score(masks_flat, preds_flat)
        precision = precision_score(masks_flat, preds_flat)
        recall = recall_score(masks_flat, preds_flat)
        f1 = f1_score(masks_flat, preds_flat)
        iou = jaccard_score(masks_flat, preds_flat)

        # 计算ROC曲线和AUC
        fpr, tpr, thresholds = roc_curve(masks_flat, probs_flat)
        roc_auc = auc(fpr, tpr)

        metrics = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "IoU": iou,
            "AUC": roc_auc
        }

        # 保存结果
        self.save_results(all_preds, all_masks, all_probs, metrics)

        # 可视化
        self.plot_confusion_matrix(masks_flat, preds_flat)
        self.plot_roc_curve(fpr, tpr, roc_auc)

        return metrics

    def generate_confusion_matrix(self, masks_flat, preds_flat):
        """生成混淆矩阵并返回统计数据"""
        cm = confusion_matrix(masks_flat, preds_flat)

        # 计算各项指标
        tn, fp, fn, tp = cm.ravel()

        confusion_matrix_dict = {
            "True Positive": tp,
            "False Positive": fp,
            "True Negative": tn,
            "False Negative": fn
        }

        return confusion_matrix_dict, cm

    def plot_confusion_matrix(self, masks_flat, preds_flat):
        """绘制并保存混淆矩阵"""
        _, cm = self.generate_confusion_matrix(masks_flat, preds_flat)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Background', 'Lung'],
                    yticklabels=['Background', 'Lung'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_roc_curve(self, fpr, tpr, roc_auc):
        """绘制并保存ROC曲线"""
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
        plt.close()

    def save_results(self, preds, masks, probs, metrics):
        """保存评估结果"""
        # 保存指标
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(self.results_dir, 'evaluation_metrics.csv'), index=False)

        # 保存预测结果示例
        sample_results = {
            "prediction": preds[0].flatten(),
            "mask": masks[0].flatten(),
            "probability": probs[0].flatten()
        }
        sample_df = pd.DataFrame(sample_results)
        sample_df.to_csv(os.path.join(self.results_dir, 'sample_predictions.csv'), index=False)

        # 保存混淆矩阵
        masks_flat = masks.flatten()
        preds_flat = preds.flatten()
        confusion_dict, cm = self.generate_confusion_matrix(masks_flat, preds_flat)
        confusion_df = pd.DataFrame([confusion_dict])
        confusion_df.to_csv(os.path.join(self.results_dir, 'confusion_matrix.csv'), index=False)

        print(f"评估结果已保存到 {self.results_dir} 目录")