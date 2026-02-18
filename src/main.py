from data_preprocessing import DataPreprocessor
from model_training import ModelTrainer
from visualization import Visualizer
from evaluation import Evaluator
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def main():
    # 1. 数据预处理
    data_path = "D:/pythonProject1/lung_nodule_project/data/Montgomery"
    preprocessor = DataPreprocessor(data_path)
    images, masks = preprocessor.load_data()
    processed_images = preprocessor.preprocess_images(images)

    # 2. 数据分割 - 保留20%作为测试集
    indices = np.arange(len(processed_images))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(processed_images))

    train_images = processed_images[:split_idx]
    train_masks = masks[:split_idx]
    test_images = processed_images[split_idx:]
    test_masks = masks[split_idx:]

    print(f"训练集大小: {len(train_images)}, 测试集大小: {len(test_images)}")

    # 训练前清理显存
    torch.cuda.empty_cache()

    # 2. 模型训练
    trainer = ModelTrainer()
    model = trainer.train_unet(train_images, train_masks, epochs=20, batch_size=2)

    # 保存模型
    torch.save(model.state_dict(), "models/unet_lung_segmentation.pth")

    # 3. 评估模型
    # 创建测试集数据加载器
    test_dataset = TensorDataset(
        torch.tensor(test_images).permute(0, 3, 1, 2).float(),
        torch.tensor(test_masks).float()
    )
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    evaluator = Evaluator(model)
    metrics = evaluator.evaluate_model(test_loader)

    print("\n模型评估结果:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # 4. 可视化界面
    visualizer = Visualizer(model, trainer)
    interface = visualizer.create_gradio_interface()
    interface.launch()


if __name__ == "__main__":
    main()