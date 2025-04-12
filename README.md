# CNN-MLP
# CIFAR-10 图像分类项目

## 项目简介

本项目使用多层感知机(MLP)和卷积神经网络(CNN)对CIFAR-10数据集进行分类，研究不同学习率和优化器对模型性能的影响，并可视化训练过程中的损失曲线和准确率曲线。

## 数据集

CIFAR-10数据集包含60,000张32x32彩色图像，分为10个类别，每个类别6,000张图像。数据集分为50,000张训练图像和10,000张测试图像。

类别包括：飞机、汽车、鸟、猫、鹿、狗、青蛙、马、船、卡车。

## 项目结构

```
cifar10-classification/
├── config.py             # 配置文件
├── data_loader.py        # 数据加载模块
├── models/
│   ├── mlp.py            # MLP模型定义
│   └── cnn.py            # CNN模型定义
├── train.py              # 训练脚本
├── test.py               # 测试脚本
├── utils.py              # 辅助工具
├── requirements.txt       # 依赖文件
└── README.md             # 项目说明
```

## 环境配置

1. 安装Python 3.7+
2. 安装依赖库：
```bash
pip install -r requirements.txt
```

## 使用方法

### 训练模型

```bash
# 训练MLP模型，使用Adam优化器，学习率0.001
python train.py --model mlp --optimizer adam --lr 0.001 --epochs 50

# 训练CNN模型，使用SGD优化器，学习率0.01
python train.py --model cnn --optimizer sgd --lr 0.01 --epochs 50
```

可选参数：
- `--model`: 模型类型 (mlp 或 cnn)
- `--optimizer`: 优化器类型 (sgd 或 adam)
- `--lr`: 学习率 (默认: 0.001)
- `--epochs`: 训练轮数 (默认: 50)
- `--batch_size`: 批量大小 (默认: 64)

### 测试模型

```bash
python test.py --model cnn --model_path ./experiments/cnn_adam_lr0.001/model.pth
```

### 结果可视化

训练过程中会自动在实验目录下生成以下文件：
- `training_curves.png`: 训练和测试的损失和准确率曲线
- TensorBoard日志: 可用于更详细的可视化
