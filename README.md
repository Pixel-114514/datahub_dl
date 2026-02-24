# Diffusion-based SR Project（教学示例）

这是一个用于**扩散超分模型**的示例项目，目标不是讲复杂原理，而是帮助你快速掌握：

- ✅ 如何搭建深度学习项目结构  
- ✅ 如何运行一个完整的训练流程  
- ✅ 如何在不乱改代码的前提下完成实验  
- ✅ 完成基于扩散模型的超分辨率模型学习

本项目基于 **PyTorch**，采用**模块化工程结构**，适合作为：
- 新人入门练习
- 课程教学示例
- 项目模板起点

---

## 教学大纲:

### 扩散超分模型实战培训大纲

| 阶段 | 课程序号 | 课程主题 | 核心内容 | 实战目标 | 建议周期（周） |
| :--- | :---: | :--- | :--- | :--- | :---: |
| **一、工程基础** | 第1课 | **深度学习项目入门** | 项目结构设计、PyTorch基础、DataLoader使用 | 搭建模块化的手写数字识别项目，告别脚本化编程 | **1 周** |
| **二、生成基础** | 第2课 | **初识生成模型 VAE** | 自编码器原理、隐变量、重参数化技巧 | 动手实现VAE，完成图像生成与隐空间插值实验 | **2 周** |
| **三、扩散核心** | 第3课 | **DDPM 原理精讲** | 前向扩散（加噪）、反向去噪原理、噪声预测 | 实现扩散过程数学公式，从零构建DDPM训练框架 | **2 周** |
|  | 第4课 | **DDPM 代码实战** | U-Net网络架构、时间步编码、采样算法 | 训练DDPM模型，实现从纯噪声生成图像 | **2 周** |
| **四、超分实战** | 第5课 | **初识超分模型** | 超分定义，扩散超分模型 SR3 | 构建超分数据集，实现CNN超分基线模型，读懂SR3论文与代码 | **1 周** |
| **五、前沿进阶** | 第6课 | **ResShift 高效超分** | 残差预测思想、加速推理机制 | 解析 ResShift 源码 | **2 周** |
## 一、项目结构说明

```text
simple_dl_project/
├── configs/                    # 配置文件（⭐新人最常修改）
│   ├── classification/         # 分类模型配置
│   │   ├── cnn.yaml           # CNN模型配置
│   │   └── resnet.yaml        # ResNet模型配置
│   └── generate/              # 生成模型配置
│       ├── ddpm.yaml          # DDPM模型配置
│       └── vae.yaml           # VAE模型配置
│
├── data/                       # 数据加载相关
│   ├── dataloader.py          # 数据加载器
│   └── MNIST/                 # MNIST数据集存储位置（自动下载）
│
├── models/                     # 模型定义
│   ├── cnn.py                 # CNN模型实现
│   ├── resnet.py              # ResNet模型实现
│   ├── vae.py                 # VAE模型实现
│   └── ddpm/                  # 扩散模型相关
│       ├── unet.py            # UNet架构实现
│       └── diffusion.py       # 扩散过程实现
│
├── trainer/                    # 训练器逻辑
│   ├── base.py                # 基础训练器类
│   ├── vae.py                 # VAE训练器类
│   └── diffusion.py           # DDPM训练器类
│
├── utils/                      # 工具函数
│   ├── logger.py              # 日志记录工具
│   └── seed.py                # 随机种子设置
│
├── checkpoints/                # 模型检查点保存目录
│   ├── cnn/
│   ├── resnet/
│   ├── vae_mnist/
│   └── ddpm_mnist/
├── main.py                     # ⭐ 程序入口（直接运行）
├── inference_vae.py            # VAE推理演示
├── vae.ipynb                   # VAE原理讲解notebook
├── ddpm_mnist.ipynb            # DDPM原理讲解notebook
├── demo.ipynb                  # 演示notebook
├── requirements.txt            # 依赖列表
└── README.md
```

### 📌 修改原则（非常重要）

| 文件 / 目录                   | 是否允许修改    |
| ----------------------------- | -------------- |
| `configs/*.yaml`              | ✅ 经常修改     |
| `models/`                     | ⚠️ 进阶后修改   |
| `trainer/`                    | ❌ 初学阶段不要改|
| `main.py`                     | ❌ 不要随意改   |


---

## 二、环境准备

### 1️⃣ 创建 Conda 环境（推荐）

```bash
conda create -n simple-dl python=3.9 -y
conda activate simple-dl
```

### 2️⃣ 安装依赖

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pyyaml
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
python main.py --config configs/classification/cnn.yaml
```

或运行VAE模型：

```bash
python main.py --config configs/generate/vae.yaml
```

或运行DDPM模型：

```bash
python main.py --config configs/generate/ddpm.yaml
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

项目包含四种模型的配置文件：
- `configs/classification/cnn.yaml` - 适用于CNN模型的配置
- `configs/classification/resnet.yaml` - 适用于ResNet模型的配置
- `configs/generate/vae.yaml` - 适用于VAE模型的配置
- `configs/generate/ddpm.yaml` - 适用于DDPM模型的配置

```yaml
# configs/generate/ddpm.yaml 示例
seed: 42                         # 随机种子

device: cuda                     # cuda / cpu

save_dir: ./checkpoints          # 模型保存目录
save_interval: 10                # 每多少个 epoch 保存一次 last.pth
exp_name: ddpm_mnist_v1          # 实验名称

train:
  epochs: 1                      # 扩散模型训练收敛较慢，建议 epoch 设大一点
  lr: 0.0002                     # 扩散模型常用学习率为 2e-4 或 5e-4
  batch_size: 128

model:
  name: ddpm                     # 模型名称，对应 MODEL_REGISTRY 中的 key
  params:
    in_channels: 1               # 输入通道数
    out_channels: 1              # 输出通道数
    model_channels: 96           # 模型基础通道数
    num_res_blocks: 2            # 每个分辨率下残差块的数量
    attention_resolutions: [7, 14] # 在 7x7 和 14x14 尺寸处使用 Attention
    channel_mult: [1, 2, 2]      # 下采样倍率控制
    dropout: 0.1                 # Dropout概率
    num_heads: 4                 # 注意力头数

diffusion:
  timesteps: 500                 # 扩散步数 (T)
  schedule: linear               # 噪声分布策略: linear 或 cosine

trainer_name: ddpm               # 训练器名称，对应 TRAINER_REGISTRY 中的 key

data:
  dataset: mnist                 # 数据集类型
  root: ./data                   # 数据集存储路径
  image_size: 28                 # 图像尺寸
  channels: 1                    # 图像通道数
```

注意：配置文件中新增了 `trainer_name` 字段，用于指定使用的训练器类型。

---

## 五、支持的模型

本项目提供了四种模型实现：

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

### 4. DDPM (`models/ddpm/unet.py` 和 `models/ddpm/diffusion.py`)
- 实现了去噪扩散概率模型(Denoising Diffusion Probabilistic Models)
- 一种先进的生成模型，通过逐步去噪过程生成高质量图像
- 包含UNet架构和扩散过程的完整实现
- 通过预测噪声或预测原图等方式进行训练

可以通过修改配置文件中的`model.name`字段来切换模型。

---

## 六、训练器功能

项目采用了**注册机制**，可以根据配置文件中的`trainer_name`字段动态选择训练器：

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

DiffusionTrainer类(`trainer/diffusion.py`)为扩散模型专门设计：
- 实现了扩散过程的前向和反向传播
- 包含噪声调度和损失计算
- 支持不同的噪声计划(Linear/Cosine)
- 提供采样和生成功能

所有训练器都注册到了`TRAINER_REGISTRY`中，可以通过配置文件中的`trainer_name`字段来选择使用哪个训练器。

训练过程中会自动保存模型到checkpoints目录下的对应子目录中。

---

## 七、数据加载

项目使用MNIST数据集进行训练和测试，数据加载器会自动下载数据集到指定目录。

- 训练集: 60,000张图像
- 测试集: 10,000张图像
- 图像尺寸: 28x28灰度图
- 类别数: 10 (数字0-9)

---

## 八、模型与训练器注册机制

为了提高代码的可扩展性，项目引入了模型和训练器的注册机制：

### 模型注册
- 在`models/__init__.py`中定义了`MODEL_REGISTRY`
- 所有模型类都注册到此字典中
- 通过配置文件中的`model.name`字段来选择模型

### 训练器注册
- 在`trainer/__init__.py`中定义了`TRAINER_REGISTRY`
- 所有训练器类都注册到此字典中
- 通过配置文件中的`trainer_name`字段来选择训练器

这种设计使得添加新模型或训练器变得非常简单，只需继承相应的基类并将其添加到注册表中即可。

---

## 九、代码整体流程

```text
读取配置
   ↓
加载数据
   ↓
从TRAINER_REGISTRY获取训练器类
   ↓
从MODEL_REGISTRY获取模型类
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

* 五、**VAE:**：阅读vae.ipynb里面索引的文章或自己寻找文章学习(https://blog.csdn.net/m0_56942491/article/details/136265500)，整理成笔记，发送至我的邮箱，以及查看model代码，了解我是如何实现vae的模型替换的。
* 六、**DDPM:**：阅读ddpm_mnist.ipynb里面的代码，回答理论、代码相关问题.整理成笔记，发送至我的邮箱，以及查看model代码，了解我是如何实现ddpm的模型替换的。



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

---

## 十二、DDPM模型

项目新增了DDPM（Denoising Diffusion Probabilistic Models）模型的支持：
- DDPM是一种强大的生成模型，能够生成高质量的图像
- 通过逐步添加噪声再逐步去噪的过程来学习数据分布
- 在`models/ddpm/`目录下包含了UNet架构和扩散过程的完整实现
- 在`trainer/diffusion.py`中实现了专门的训练器
- 在`ddpm_mnist.ipynb`中提供了详细的原理讲解和示例

要运行DDPM模型，可以使用：

```bash
python main.py --config configs/generate/ddpm.yaml
```

这将训练一个在MNIST数据集上的DDPM模型。