# Simple Diffusion SR Project

这是一个面向培训的 PyTorch 教学项目，目标是用尽量清晰的工程结构，把以下几条主线串起来：

- CNN / ResNet 分类训练
- VAE 生成建模
- DDPM 基础扩散
- 超分辨率基线训练
- ResShift 风格的扩散式超分教学实现

当前仓库里的 `ResShift` 是教学版实现，重点是帮助学员理解“残差迁移 + 少步数采样”的核心思路，不是官方仓库的完整复现。

## 当前支持的训练入口

```bash
python main.py --config configs/classification/cnn.yaml
python main.py --config configs/classification/resnet.yaml
python main.py --config configs/generate/vae.yaml
python main.py --config configs/generate/ddpm.yaml
python main.py --config configs/sr/srresnet.yaml
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
│   └── dataloader.py
├── docs/
│   └── resshift.md
├── models/
│   ├── ddpm/
│   ├── cnn.py
│   ├── resnet.py
│   ├── sr.py
│   ├── resshift.py
│   └── vae.py
├── trainer/
│   ├── base.py
│   ├── diffusion.py
│   ├── sr.py
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
- 增加了教学版 `ResShiftUNet + ResidualShiftScheduler`
- 增加了 PSNR 评估与超分可视化保存
- 增加了 `configs/sr/` 配置和 `docs/resshift.md` 学习材料

## 超分任务现在怎么跑

超分训练不再直接复用分类标签，而是走一条单独的数据流：

1. 从 `MNIST / FashionMNIST / CIFAR10` 读取高分辨率图像
2. 通过下采样构造低分辨率图像
3. 再把低分辨率图像插值回原尺寸
4. 返回 `(lr_up, hr)` 这一对样本给训练器

这样做的好处是，学员不需要先准备 DIV2K 这类大数据集，就能把超分训练流程完整跑通。

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
- 适合教学阶段：先理解超分任务、PSNR 和残差学习

### 2. ResShift 教学版 `configs/sr/resshift.yaml`

- 模型文件：`models/resshift.py`
- 训练器：`trainer/resshift.py`
- 思路：把高分图和低分条件图之间的残差拆进一个少步数的“shift”过程
- 训练目标：随机采样时间步，预测 `HR - LR_up` 残差
- 推理过程：从低清条件图加噪开始，逐步恢复残差并重建高分图

这版实现保留了 ResShift 最适合培训讲解的几个关键词：

- 条件输入不是纯噪声，而是退化后的低质图像
- 关注的是“残差迁移”而不是无条件生成
- 推理步数明显比传统 DDPM 更短

但它没有追求官方仓库级别的真实图像恢复效果，训练数据、退化方式、网络规模都做了教学化简化。

## 输出内容

- 分类任务：保存 `last.pth / best.pth`
- VAE / DDPM：保存权重与生成样例
- SR / ResShift：额外保存 `sr_epoch_x.png`

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

## 推荐学习顺序

1. 先跑 `cnn.yaml` 和 `resnet.yaml`，理解通用训练入口
2. 再跑 `vae.yaml`，理解生成模型和 trainer 定制
3. 再跑 `ddpm.yaml`，理解时间步、噪声调度和采样
4. 然后跑 `srresnet.yaml`，建立超分任务直觉
5. 最后跑 `resshift.yaml`，理解条件扩散超分和残差迁移

## ResShift 资料

- 仓库内讲义：`docs/resshift.md`
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
