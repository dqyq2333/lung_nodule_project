# 肺部X光图像分割系统

基于 UNet 的胸部X光图像肺部分割项目，集成 LLM 自动生成医学报告。

## 技术框架

- **深度学习**: PyTorch
- **图像处理**: OpenCV, scikit-image
- **数值计算**: NumPy
- **机器学习**: scikit-learn
- **可视化**: Matplotlib, Seaborn
- **数据处理**: Pandas
- **Web界面**: Gradio
- **LLM推理**: llama-cpp-python

## 功能特性

- 胸部X光图像肺部区域自动分割
- UNet 编码器-解码器架构，支持 Skip Connection 特征融合
- 多指标评估体系（Accuracy, Precision, Recall, F1, IoU, AUC）
- 混淆矩阵与 ROC 曲线可视化
- Gradio 交互式 Web 界面
- 集成 DeepSeek-R1 模型自动生成医学报告

## 项目结构

```
lung_nodule_project/
├── data/
│   └── Montgomery/              # Montgomery County 胸部X光数据集
│       ├── CXR_png/             # 原始X光图像
│       ├── ClinicalReadings/    # 临床诊断报告
│       └── ManualMask/          # 手工标注肺部掩码
│           ├── leftMask/        # 左肺掩码
│           └── rightMask/       # 右肺掩码
│
└── src/
    ├── main.py                  # 主程序入口
    ├── data_preprocessing.py    # 数据预处理模块
    ├── model_training.py        # UNet模型与训练
    ├── evaluation.py            # 模型评估模块
    ├── visualization.py         # Gradio界面与LLM报告生成
    ├── models/                  # 保存的模型权重
    ├── results/                 # 评估结果
    └── predictions/             # 预测结果图像
```

## 安装说明

### 环境要求

- Python 3.8+
- CUDA 11.0+ (GPU加速)

### 安装依赖

```bash
pip install torch torchvision
pip install opencv-python
pip install numpy scikit-learn scikit-image
pip install matplotlib seaborn pandas
pip install gradio
pip install llama-cpp-python
```

## 使用方法

### 1. 数据准备

将 Montgomery 数据集放置于 `data/Montgomery/` 目录下，包含：
- `CXR_png/`: 胸部X光图像
- `ManualMask/leftMask/`: 左肺掩码
- `ManualMask/rightMask/`: 右肺掩码

### 2. 训练模型

```bash
cd src
python main.py
```

训练参数：
- Epochs: 20
- Batch Size: 2
- Learning Rate: 0.001
- Optimizer: Adam
- Loss: BCELoss

### 3. 启动Web界面

```bash
cd src
python visualization.py
```

访问 Gradio 界面上传X光图像进行预测和报告生成。

## 模型架构

```
UNet:
├── 编码器: 4层卷积下采样 (32→64→128→256通道)
├── 瓶颈层: 512通道
├── 解码器: 4层转置卷积上采样 + Skip Connection
└── 输出层: Sigmoid激活
```

特点：
- BatchNorm 加速收敛
- Skip Connection 特征融合
- center_crop 尺寸对齐

## 模型性能

| 指标 | 数值 |
|------|------|
| Accuracy | 98.26% |
| Precision | 96.83% |
| Recall | 96.28% |
| F1 Score | 96.55% |
| IoU | 93.34% |
| AUC | 99.75% |

## 数据预处理流程

1. 图像加载与缩放至 512×512
2. 直方图均衡化增强对比度
3. 高斯模糊降噪 (5×5核)
4. 归一化至 [0, 1]
5. 左右肺掩码合并与二值化

## 优化亮点

- UNet架构引入 BatchNorm 加速收敛与 Skip Connection 特征融合
- 采用 ReduceLROnPlateau 自适应学习率调度策略
- 多维度评估体系（IoU, Dice, AUC等）
- 集成 LLM 实现医学报告自动生成

## 数据集

使用 Montgomery County 胸部X光数据集，包含138张胸部X光图像及对应的手工标注肺部掩码。

## 许可证

MIT License
