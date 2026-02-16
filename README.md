# Simple Deep Learning Project（教学示例）

这是一个用于**深度学习新人入门**的示例项目，目标不是讲复杂原理，而是帮助你快速掌握：

- ✅ 如何搭建深度学习项目结构  
- ✅ 如何运行一个完整的训练流程  
- ✅ 如何在不乱改代码的前提下完成实验  

本项目基于 **PyTorch**，采用**模块化工程结构**，适合作为：
- 新人入门练习
- 课程教学示例
- 项目模板起点

---

## 一、项目结构说明

```text
simple_dl_project/
├── configs/                # 配置文件（⭐新人最常修改）
│   ├── cnn.yaml           # CNN模型配置
│   ├── resnet.yaml        # ResNet模型配置
│   └── vae.yaml           # VAE模型配置
│
├── data/                   # 数据加载相关
│   ├── dataloader.py      # 数据加载器
│   └── MNIST/             # MNIST数据集存储位置（自动下载）
│
├── models/                 # 模型定义
│   ├── cnn.py             # CNN模型实现
│   ├── resnet.py          # ResNet模型实现
│   └── vae.py             # VAE模型实现
│
├── trainer/                # 训练器逻辑
│   ├── base.py            # 基础训练器类
│   └── vae.py             # VAE训练器类
│
├── utils/                  # 工具函数
│   ├── logger.py          # 日志记录工具
│   └── seed.py            # 随机种子设置
│
├── checkpoints/            # 模型检查点保存目录
│   ├── cnn/
│   ├── resnet/
│   └── vae_mnist/
├── main.py                 # ⭐ 程序入口（直接运行）
├── inference_vae.py        # VAE推理演示
├── vae.ipynb               # VAE原理讲解notebook
├── demo.ipynb              # 演示notebook
├── requirements.txt        # 依赖列表
└── README.md
```

### 📌 修改原则（非常重要）

| 文件 / 目录               | 是否允许修改    |
| ------------------------- | -------------- |
| `configs/*.yaml`          | ✅ 经常修改     |
| `models/`                 | ⚠️ 进阶后修改   |
| `trainer/`                | ❌ 初学阶段不要改|
| `main.py`                 | ❌ 不要随意改   |


---

## 二、环境准备

### 1️⃣ 创建 Conda 环境（推荐）

```bash
conda create -n simple-dl python=3.9 -y
conda activate simple-dl
```

### 2️⃣ 安装依赖

```bash
自行完成GPU版本的PyTorch安装
```

依赖列表 (`requirements.txt`)：
- torch
- torchvision
- pyyaml

> 如果你有 GPU 且安装了 CUDA，请确保 PyTorch 支持 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 三、如何运行项目

在项目根目录下执行：

```bash
python main.py --config configs/cnn.yaml
```

或运行VAE模型：

```bash
python main.py --config configs/vae.yaml
```

正常情况下你将看到类似输出：

```text
PyTorch 版本: 2.x.x
CUDA 是否可用: True
CUDA 版本 (torch 编译时用的): 11.8
当前 GPU 名称: GeForce RTX xx
Epoch [1] | Train Loss: 0.35 | Val Acc: 0.90
Epoch [2] | Train Loss: 0.21 | Val Acc: 0.94
...
```

这说明：

* 数据加载成功
* 模型正常训练
* 验证流程正常

---

## 四、配置文件说明

项目包含三个配置文件：
- `configs/cnn.yaml` - 适用于CNN模型的配置
- `configs/resnet.yaml` - 适用于ResNet模型的配置
- `configs/vae.yaml` - 适用于VAE模型的配置

```yaml
# configs/resnet.yaml 示例
seed: 42                 # 随机种子

device: cuda             # cuda / cpu

save_dir: ./checkpoints  # 模型保存目录
save_interval: 5         # 每多少个 epoch 保存一次 last.pth
exp_name: resnet         # 实验名称，留空会自动生成时间戳

train:
  epochs: 10
  lr: 0.001
  batch_size: 128

model:
  name: resnet           # 模型名称 (cnn 或 resnet)
  params:
    num_classes: 10      # 分类数量

data:
  root: ./data           # 数据集存储路径
```



---

## 五、支持的模型

本项目提供了三种模型实现：

### 1. SimpleCNN (`models/cnn.py`)
- 一个简单但效果不错的CNN，适合MNIST手写数字分类
- 参数少、训练快、测试准确率轻松99%+
- 包含BatchNorm、多种激活函数选项(gelu/relu/silu)等现代设计

### 2. ResNet (`models/resnet.py`)
- 实现了基本的ResNet架构，包含残差块
- 适合学习残差网络的设计思想
- 包含shortcut连接以解决梯度消失问题

### 3. VAE (`models/vae.py`)
- 实现了卷积变分自编码器(Convolutional Variational Autoencoder)
- 用于学习数据分布并生成新样本
- 包含编码器(Encoder)、解码器(Decoder)和重参数化技巧(Reparameterization Trick)
- 通过最小化重构损失和KL散度来训练

可以通过修改配置文件中的`model.name`字段来切换模型。

---

## 六、训练器功能

BaseTrainer类(`trainer/base.py`)提供以下功能：
- 自动检测设备(CPU/GPU)
- 模型训练和验证
- 检查点保存(包括最佳模型和最新模型)
- 日志记录
- 实验配置保存

VAETrainer类(`trainer/vae.py`)专门针对VAE模型的特点进行了定制：
- 实现了VAE特有的损失函数(重构损失+KL散度)
- 提供了专门的训练和验证方法
- 支持重构和生成任务

训练过程中会自动保存模型到checkpoints目录下的对应子目录中。

---

## 七、数据加载

项目使用MNIST数据集进行训练和测试，数据加载器会自动下载数据集到指定目录。

- 训练集: 60,000张图像
- 测试集: 10,000张图像
- 图像尺寸: 28x28灰度图
- 类别数: 10 (数字0-9)

---

## 八、VAE模型详解

VAE(Variational Autoencoder，变分自编码器)是一种生成模型，它学习将输入数据映射到一个潜在空间(latent space)，然后从这个潜在空间生成新的数据。

### VAE的核心概念：
1. **编码器(Encoder)**：将输入图像压缩为潜在空间的分布参数(均值μ和方差σ²)
2. **重参数化技巧(Reparameterization Trick)**：使得模型可微分，能够进行端到端训练
3. **解码器(Decoder)**：将潜在空间的样本转换回原始数据空间
4. **损失函数**：由重构损失和KL散度组成，平衡了重构质量和潜在空间的正则化

### 代码实现：
- 在`models/vae.py`中实现了ConvVAE类
- 在`trainer/vae.py`中实现了VAETrainer类
- 在`vae.ipynb`中提供了详细的原理讲解和示例

---

## 九、代码整体流程

```text
读取配置
   ↓
加载数据
   ↓
创建模型
   ↓
定义 loss & optimizer
   ↓
for epoch:
    训练一轮
    验证准确率
    保存检查点(如果需要)
```

你只需要记住一句话：

> **main.py 负责"调度"，其他模块负责"干活"。**

---

## 十、作业：
* 一、将代码跑通,自己处理报错

* 二、在main.py中添加一个函数，可以将一些样本展示出来（提示：使用matplotlib.pyplot）

* 二、将代码过一遍，要能够说明每个模块的作用是什么，以及一个深度学习模型的训练流程

* 三、models文件夹里面有两个模型，我们训练用的是SimpleCNN，你需要观察代码，调用ResNet模型（提示：观察trainer/base.py中的模型创建部分）。相关参数在configs/resnet.yaml中。

* 四、模型训练完后，会在checkpoints目录下保存模型训练文件（包括模型参数、优化器状态、训练配置等）。你需要写一个函数，调用模型推理（inference），输入一张图片，输出模型的预测结果。

* 五、阅读 vae.ipynb 里面索引的文章 [《变分自编码器 VAE 超详解》](https://blog.csdn.net/m0_56942491/article/details/136265500),也可以自行寻找相关文章,将学习的内容和遇到的问题用md记录,发送至我的邮箱
  
* 六、查看model代码，了解我是如何实现vae的模型替换的。


通过这些练习，你将逐步熟悉深度学习项目的各个组成部分及其相互关系。

---

## 十一、VAE推理演示

项目还包含了VAE推理演示脚本`inference_vae.py`，可以用来：
- 加载训练好的VAE模型
- 对测试图像进行重构
- 从潜在空间随机生成新样本

运行以下命令执行推理：

```bash
python inference_vae.py
```

这将在`inference_results`目录下生成重构图像和生成样本的图片。
