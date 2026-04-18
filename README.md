# Simple Diffusion SR Project

这是一个 PyTorch 项目，目标是用尽量清晰的工程结构，把以下几条主线串起来：

- CNN / ResNet 分类训练
- VAE 生成建模
- DDPM 基础扩散
- 超分辨率基线训练
- SR3 条件扩散超分实现
- ResShift 风格的扩散式超分实现

当前仓库里的 `ResShift` 是简化实现，重点是保留“残差迁移 + 少步数采样”的核心思路，不是官方仓库的完整复现。

## 阅读入口

第一次看这个项目时，先不要直接扎进某个模型文件。

先按下面顺序读：

1. `docs/architecture.md`
2. `docs/learning_path.md`
3. `docs/generative_basics.md`
4. `docs/mit_6s184_flow_matching_notes.md`
5. `configs/classification/cnn.yaml`
6. `main.py`
7. `trainer/base.py`
8. 再回头看具体任务的 `trainer` 和 `model`

`docs/architecture.md` 讲项目分层，`docs/learning_path.md` 讲整体顺序，`docs/generative_basics.md` 讲 VAE、DDPM、SR3、ResShift 这些知识点，`docs/mit_6s184_flow_matching_notes.md` 把 MIT 6.S184 课程内容映射到本仓库代码。

## 当前支持的训练入口

```bash
python main.py --config configs/classification/cnn.yaml
python main.py --config configs/classification/resnet.yaml
python main.py --config configs/generate/vae.yaml
python main.py --config configs/generate/ddpm.yaml
python main.py --config configs/sr/srresnet.yaml
python main.py --config configs/sr/sr3.yaml
python main.py --config configs/sr/resshift.yaml
```

如果没有可用 GPU，可以直接加：

```bash
python main.py --config configs/sr/resshift.yaml --device cpu
```

## 项目结构

```text
simple_dl_project/
├── configs/
│   ├── classification/
│   ├── generate/
│   └── sr/
├── data/
│   ├── dataloader.py
│   ├── sr_dataset.py
│   └── transforms.py
├── docs/
│   ├── architecture.md
│   ├── learning_path.md
│   ├── generative_basics.md
│   ├── mit_6s184_flow_matching_notes.md
│   └── resshift.md
├── models/
│   ├── ddpm/
│   ├── cnn.py
│   ├── resnet.py
│   ├── sr.py
│   ├── sr3.py
│   ├── resshift.py
│   └── vae.py
├── trainer/
│   ├── base.py
│   ├── diffusion.py
│   ├── sr.py
│   ├── sr3.py
│   ├── resshift.py
│   └── vae.py
├── utils/
│   ├── logger.py
│   ├── metrics.py
│   └── seed.py
├── demo.ipynb
├── ddpm_mnist.ipynb
├── inference_vae.py
├── main.py
├── sr.ipynb
└── vae.ipynb
```

## 这次完善了什么

- 修正了训练入口默认配置路径，并补上随机种子设置
- 扩展了 `data/dataloader.py`，支持分类、生成和超分任务
- 增加了 `ResNet` 注册，修复了 README 说支持但代码未注册的问题
- 给 DDPM 加上了 `cosine schedule` 和更合理的 Attention 分辨率判断
- 增加了超分基线 `SimpleSRResNet`
- 增加了简化版 `ResShiftUNet + ResidualShiftScheduler`
- 增加了简化版 `SR3UNet + SR3Trainer`，补上 DDPM 到 ResShift 的桥梁
- 增加了 PSNR 评估与超分可视化保存
- 增加了 `configs/sr/` 配置和 `docs/resshift.md` 学习材料
- 把 trainer 的 best/save 逻辑统一成通用“监控指标”机制
- 修复了 DDPM 依赖伪指标保存 checkpoint 的问题，改为记录 `Val Noise Loss`
- 把数据层里最容易单独阅读的部分拆成 `data/transforms.py` 和 `data/sr_dataset.py`
- 增加了 `docs/architecture.md`，专门解释项目为什么这样分层
- 增加了 `docs/learning_path.md`，把顺序改成“分类 -> VAE -> DDPM -> SR 基线 -> SR3 -> ResShift”
- 增加了 `docs/generative_basics.md`，专门补充 VAE、DDPM、SR3、ResShift 的知识点

## 先理解什么，再理解什么

这不是一个“先看网络细节”的项目，而是一个“先看工程骨架，再看算法差异”的项目。

建议顺序是：

1. 先理解 `main.py -> trainer -> model -> data` 这条主链路
2. 再从分类任务进入生成任务
3. 再从无条件扩散进入条件扩散超分
4. 最后再去理解 ResShift 这种面向恢复任务的进一步改造

如果一开始就钻进 `UNet` 或 `ResNet` 细节，通常会看懂局部，看不懂整体。

## 生成模型知识从哪里看

如果你会问下面这些问题：

- VAE 为什么要有 `mu / logvar`
- KL 散度到底在约束什么
- 扩散模型为什么要预测噪声
- DDPM 为什么会慢
- SR3 为什么是 DDPM 到 ResShift 的桥梁
- ResShift 和普通 DDPM 到底差在哪

推荐直接看：

- `docs/learning_path.md`
- `docs/generative_basics.md`
- `docs/mit_6s184_flow_matching_notes.md`
- `docs/resshift.md`

这几份文档是按顺序写的，不要求先掌握完整论文公式。

## 超分任务现在怎么跑

超分训练不再直接复用分类标签，而是走一条单独的数据流：

1. 从 `MNIST / FashionMNIST / CIFAR10` 读取高分辨率图像
2. 通过下采样构造低分辨率图像
3. 再把低分辨率图像插值回原尺寸
4. 返回 `(lr_up, hr)` 这一对样本给训练器

这样做的好处是，不需要先准备 DIV2K 这类大数据集，就能把超分训练流程完整跑通。

如果想加速实验，可以在配置里加：

```yaml
data:
  max_train_samples: 1024
  max_val_samples: 256
```

## 两条超分主线

### 1. 超分基线 `configs/sr/srresnet.yaml`

- 模型文件：`models/sr.py`
- 训练器：`trainer/sr.py`
- 思路：输入双三次插值后的低清图，直接预测高频残差
- 适合先建立超分任务、PSNR 和残差学习的直觉

### 2. SR3 简化版 `configs/sr/sr3.yaml`

- 模型文件：`models/sr3.py`
- 训练器：`trainer/sr3.py`
- 思路：把 DDPM 变成带低清条件图的扩散超分模型
- 适合作为 DDPM 到 ResShift 的桥梁，先理解“条件扩散超分”

### 3. ResShift 简化版 `configs/sr/resshift.yaml`

- 模型文件：`models/resshift.py`
- 训练器：`trainer/resshift.py`
- 思路：把高分图和低分条件图之间的残差拆进一个少步数的“shift”过程
- 训练目标：随机采样时间步，预测 `HR - LR_up` 残差
- 推理过程：从低清条件图加噪开始，逐步恢复残差并重建高分图

这版实现保留了 ResShift 最核心的几个关键词：

- 条件输入不是纯噪声，而是退化后的低质图像
- 关注的是“残差迁移”而不是无条件生成
- 推理步数明显比传统 DDPM 更短

但它没有追求官方仓库级别的真实图像恢复效果，训练数据、退化方式、网络规模都做了简化。

## 输出内容

- 分类任务：保存 `last.pth / best.pth`
- VAE / DDPM：保存权重与生成样例
- SR / SR3 / ResShift：额外保存 `sr_epoch_x.png`

`sr_epoch_x.png` 的三行含义分别是：

- 第一行：低清输入
- 第二行：模型输出
- 第三行：高分真值

## 配置约定

常用字段如下：

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

`data.value_range` 目前支持：

- `zero_one`
- `minus_one_one`

## trainer 层现在怎么理解

这次更新后，`trainer/base.py` 可以从一个更统一的角度去理解：

- 它不再默认只服务分类任务
- 每个任务只需要告诉基类“我监控什么指标”
- 基类统一负责比较 best、保存 `best.pth` 和 `last.pth`

这里最值得注意的是：

工程里最怕的不是没有抽象，而是多个任务各写一套相似但不一致的流程。

这次重构就是一个很典型的“把重复逻辑收回父类”的例子。

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
- 仓库内讲义：`docs/resshift.md`
  - 已补充 `SR3 vs ResShift` 对比
  - 已补充从 `SR3` 改造成 `ResShift` 的实现路径
  - 已补充练习题与论文公式复现题
- ResShift 论文：<https://arxiv.org/abs/2307.12348>
- ResShift 官方仓库：<https://github.com/zsyOAOA/ResShift>
- SR3 论文：<https://arxiv.org/abs/2104.07636>

## 依赖

```bash
pip install -r requirements.txt
```

当前 `requirements.txt` 包含：

- `torch`
- `torchvision`
- `pyyaml`
- `numpy`
- `matplotlib`
