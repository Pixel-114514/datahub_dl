# Simple Diffusion SR Project

## 30 秒快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 跑一个分类任务（确认环境没问题）
python main.py --config configs/classification/cnn.yaml

# 3. 跑 ResShift 超分（没有 GPU 加 --device cpu）
python main.py --config configs/sr/resshift.yaml
```

> 跑通上面两步后，再看下面的详细内容。

> DDPM 和 SR3 完整训练需要较长时间（详见 `docs/learning_path.md` 各阶段预估时间）。想快速验证流程，可以在 yaml 里加 `data.max_train_samples: 512` 和 `data.max_val_samples: 128` 减少数据量。

---

这是一个 PyTorch 项目，用尽量清晰的工程结构串起几条主线：

- CNN / ResNet 分类训练
- VAE 生成建模
- DDPM 基础扩散
- 超分辨率基线训练
- SR3 条件扩散超分
- ResShift 风格的扩散式超分

仓库里的 ResShift 是简化实现，保留"残差迁移 + 少步数采样"的核心思路，不是官方仓库的完整复现。

## 阅读入口

第一次看这个项目，先别直接扎进某个模型文件。按这个顺序读：

1. `docs/architecture.md` — 项目分层
2. `docs/learning_path.md` — 整体顺序
3. `docs/generative_basics.md` — VAE、DDPM、SR3、ResShift 的知识点
4. `docs/mit_6s184_flow_matching_notes.md` — MIT 课程与仓库代码的对照
5. `configs/classification/cnn.yaml` — 看配置长什么样
6. `main.py` — 看配置怎么被读取
7. `trainer/base.py` — 看通用训练骨架
8. 再回头看具体任务的 trainer 和 model

## 训练入口

```bash
python main.py --config configs/classification/cnn.yaml
python main.py --config configs/classification/resnet.yaml
python main.py --config configs/generate/vae.yaml
python main.py --config configs/generate/ddpm.yaml
python main.py --config configs/sr/srresnet.yaml
python main.py --config configs/sr/sr3.yaml
python main.py --config configs/sr/resshift.yaml
```

没有 GPU 加 `--device cpu`：

```bash
python main.py --config configs/sr/resshift.yaml --device cpu
```

## 项目结构

```text
simple_dl_project/
├── configs/
│   ├── classification/       # cnn.yaml, resnet.yaml
│   ├── generate/             # vae.yaml, ddpm.yaml
│   └── sr/                   # srresnet.yaml, sr3.yaml, resshift.yaml
├── data/
│   ├── dataloader.py         # 统一数据加载入口
│   ├── sr_dataset.py         # 超分数据包装
│   └── transforms.py         # 图像预处理
├── docs/
│   ├── architecture.md       # 项目分层和调用链
│   ├── learning_path.md      # 学习路径
│   ├── generative_basics.md  # 生成模型核心概念
│   ├── mit_6s184_flow_matching_notes.md  # MIT 课程对照
│   └── resshift.md           # ResShift 学习说明
├── models/
│   ├── ddpm/
│   │   ├── unet.py           # UNet 扩散去噪网络
│   │   └── diffusion.py      # GaussianDiffusion 调度器
│   ├── cnn.py, resnet.py     # 分类网络
│   ├── vae.py                # ConvVAE
│   ├── sr.py                 # SimpleSRResNet
│   ├── sr3.py                # SR3UNet
│   └── resshift.py           # ResShiftUNet + ResidualShiftScheduler
├── trainer/
│   ├── base.py               # BaseTrainer 通用训练骨架
│   ├── diffusion.py          # DiffusionTrainer
│   ├── sr.py                 # BaseSRTrainer + SuperResolutionTrainer
│   ├── sr3.py                # SR3Trainer
│   ├── resshift.py           # ResShiftTrainer
│   └── vae.py                # VAETrainer
├── utils/
│   ├── logger.py, metrics.py, seed.py
├── demo.ipynb, ddpm_mnist.ipynb, sr.ipynb, vae.ipynb
├── inference_vae.py
└── main.py
```

## 先理解什么再理解什么

这个项目不是"先看网络细节"，而是"先看工程骨架，再看算法差异"。

建议顺序：

1. 先理解 `main.py → trainer → model → data` 这条主链路
2. 再从分类任务进入生成任务
3. 再从无条件扩散进入条件扩散超分
4. 最后理解 ResShift 这种面向恢复任务的改造

一开始就钻进 UNet 或 ResNet 细节，通常会看懂局部、看不懂整体。

## 生成模型知识从哪里看

如果你会问这些问题：

- VAE 为什么要有 `mu / logvar`
- KL 散度到底在约束什么
- 扩散模型为什么要预测噪声
- DDPM 为什么会慢
- SR3 为什么是 DDPM 到 ResShift 的桥梁
- ResShift 和普通 DDPM 到底差在哪

推荐直接看 `docs/learning_path.md`、`docs/generative_basics.md`、`docs/resshift.md`。这几份文档按顺序写，不要求先掌握完整论文公式。

## 超分任务怎么跑

超分训练不直接复用分类标签，走一条单独的数据流：

1. 从 MNIST / FashionMNIST / CIFAR10 读取高分辨率图像
2. 下采样构造低分辨率图像
3. 把低分辨率图像插值回原尺寸
4. 返回 `(lr_up, hr)` 给训练器

不需要先准备 DIV2K 这类大数据集，就能把超分训练流程完整跑通。

加速实验：

```yaml
data:
  max_train_samples: 1024
  max_val_samples: 256
```

## 两条超分主线

### 1. 超分基线 `configs/sr/srresnet.yaml`

输入双三次插值后的低清图，直接预测高频残差。适合先建立超分任务、PSNR 和残差学习的直觉。

### 2. SR3 简化版 `configs/sr/sr3.yaml`

把 DDPM 变成带低清条件图的扩散超分模型。适合作为 DDPM 到 ResShift 的桥梁，先理解"条件扩散超分"。

### 3. ResShift 简化版 `configs/sr/resshift.yaml`

把高分图和低分条件图之间的残差拆进一个少步数的"shift"过程。训练目标是随机采样时间步，预测 `HR - LR_up` 残差。推理从低清条件图加噪开始，逐步恢复残差并重建高分图。

保留了 ResShift 最核心的几个点：

- 条件输入不是纯噪声，而是退化后的低质图像
- 关注"残差迁移"而不是无条件生成
- 推理步数明显比传统 DDPM 更短

但没有追求官方仓库级别的真实图像恢复效果，训练数据、退化方式、网络规模都做了简化。

## 输出内容

- 分类任务：保存 `last.pth / best.pth`
- VAE / DDPM：保存权重与生成样例
- SR / SR3 / ResShift：额外保存 `sr_epoch_x.png`

`sr_epoch_x.png` 三行分别是：低清输入、模型输出、高分真值。

## 配置约定

```yaml
train:
  epochs: 10
  lr: 0.0002
  batch_size: 128

model:
  name: resshift
  params:
    model_channels: 64

trainer_name: resshift

data:
  task: super_resolution
  dataset: mnist
  root: ./data
  image_size: 28
  scale_factor: 2
  value_range: zero_one
```

`data.value_range` 支持 `zero_one` 和 `minus_one_one`。

## trainer 层怎么理解

`trainer/base.py` 不再默认只服务分类任务。每个任务告诉基类"我监控什么指标"，基类统一负责比较 best、保存 `best.pth` 和 `last.pth`。

工程里最怕的不是没有抽象，而是多个任务各写一套相似但不一致的流程。这次重构就是"把重复逻辑收回父类"的典型例子。

## 推荐学习顺序

1. 先跑 `cnn.yaml` 和 `resnet.yaml`，理解通用训练入口
2. 再跑 `vae.yaml`，理解生成模型和 trainer 定制
3. 再跑 `ddpm.yaml`，理解时间步、噪声调度和采样
4. 然后跑 `srresnet.yaml`，建立超分任务直觉
5. 再跑 `sr3.yaml`，理解条件扩散超分
6. 最后跑 `resshift.yaml`，理解少步扩散超分和残差迁移

## ResShift 资料

- 生成模型基础讲义：`docs/generative_basics.md`
- 学习路径：`docs/learning_path.md`
- 项目架构讲义：`docs/architecture.md`
- 仓库内讲义：`docs/resshift.md`（含 SR3 vs ResShift 对比、改造路线、练习题）
- ResShift 论文：<https://arxiv.org/abs/2307.12348>
- ResShift 官方仓库：<https://github.com/zsyOAOA/ResShift>
- SR3 论文：<https://arxiv.org/abs/2104.07636>

## 依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt`：`torch`、`torchvision`、`pyyaml`、`numpy`、`matplotlib`。
