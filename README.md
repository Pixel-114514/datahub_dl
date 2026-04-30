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

> DDPM 和 SR3 完整训练需要较长时间（详见 [学习路径](docs/learning_path.md) 各阶段预估时间）。想快速验证流程，可以在 yaml 里加 `data.max_train_samples: 512` 和 `data.max_val_samples: 128` 减少数据量。

---

这是一个 PyTorch 教学项目，用尽量清晰的工程结构串起六条学习主线：

| 主线 | 任务类型 | 核心概念 |
|------|----------|----------|
| CNN / ResNet | 分类 | 训练循环、loss、优化器 |
| VAE | 生成 | 潜变量、KL 散度、重参数化 |
| DDPM | 生成 | 时间步、噪声预测、多步采样 |
| SRResNet | 超分 | 残差学习、PSNR、数据包装 |
| SR3 | 超分 | 条件扩散、通道拼接 |
| ResShift | 超分 | 残差迁移、少步采样 |

仓库里的 ResShift 是简化实现，保留"残差迁移 + 少步数采样"的核心思路，不是官方仓库的完整复现。

## 教学路线安排

这条路径的设计原则是**每一步只变一个东西**：

```
分类（学会训练）
 → VAE（从判别到生成，引入潜变量）
  → DDPM（从一步到位到多步迭代，引入时间步）
   → SRResNet（从生成到恢复，引入残差学习）
    → SR3（无条件扩散 → 条件扩散，引入条件输入）
     → ResShift（预测噪声 → 预测残差，引入少步采样）
```

为什么这样排：

- **分类 → VAE**：从"做判断"转到"建模分布"，先搞懂潜变量和 loss 由多项组成
- **VAE → DDPM**：从"一步编解码"转到"多步加噪去噪"，理解时间步和噪声调度
- **DDPM → SRResNet**：从"无条件生成"转到"条件恢复"，理解残差学习和 PSNR
- **SRResNet → SR3**：只变一件事——把低清条件图拼到输入里，网络结构不变
- **SR3 → ResShift**：只变一件事——预测目标从噪声换成残差，采样起点从纯噪声换成低清图附近

每一步之间的跨度都控制在一个核心概念以内，不会出现"从 A 直接跳到 D"的情况。

详细的学习路径（每阶段跑什么、看什么代码、预期输出、常见问题）见 [docs/learning_path.md](docs/learning_path.md)。

## 文档导航

| 文档 | 适合谁 | 讲什么 |
|------|--------|--------|
| [项目架构导读](docs/architecture.md) | 第一次来的学员 | 四层结构（config / main / trainer / model / data）、调用链、注册表模式 |
| [学习路径](docs/learning_path.md) | 跟着走的学员 | 七个阶段的具体操作、代码指引、预期输出 |
| [生成模型知识补充](docs/generative_basics.md) | 跑通代码后想搞懂原理的学员 | VAE、DDPM、SR3、ResShift 的核心概念和代码对应 |
| [ResShift 学习说明](docs/resshift.md) | 学到 ResShift 阶段的学员 | 算法详解、和 SR3 的对比、从 SR3 改造的路线、练习题 |
| [MIT 课程对照笔记](docs/mit_6s184_flow_matching_notes.md) | 想深入理论的学员 | MIT 6.S184 课程与仓库代码的映射，flow matching / score matching |

建议按上面的顺序读。先看架构建立全局认知，再按学习路径走，遇到不懂的概念查 generative_basics，学到 ResShift 时看 resshift.md，想补理论看 MIT 笔记。

## 训练入口

```bash
python main.py --config configs/classification/cnn.yaml       # CNN 分类
python main.py --config configs/classification/resnet.yaml     # ResNet 分类
python main.py --config configs/generate/vae.yaml              # VAE 生成
python main.py --config configs/generate/ddpm.yaml             # DDPM 扩散
python main.py --config configs/sr/srresnet.yaml               # 超分基线
python main.py --config configs/sr/sr3.yaml                    # SR3 条件扩散超分
python main.py --config configs/sr/resshift.yaml               # ResShift 少步超分
```

没有 GPU 加 `--device cpu`。每个配置的详细说明见 [学习路径](docs/learning_path.md) 对应阶段。

## 项目结构

```text
simple_dl_project/
├── configs/                        # 实验配置
│   ├── classification/             #   cnn.yaml, resnet.yaml
│   ├── generate/                   #   vae.yaml, ddpm.yaml
│   └── sr/                         #   srresnet.yaml, sr3.yaml, resshift.yaml
├── data/
│   ├── dataloader.py               # 统一数据加载入口
│   ├── sr_dataset.py               # 超分数据包装：(image,label) → (lr_up,hr)
│   └── transforms.py               # 图像预处理和值域映射
├── docs/                           # 教学文档（见上方导航表）
├── models/
│   ├── ddpm/
│   │   ├── unet.py                 # UNet 扩散去噪网络
│   │   └── diffusion.py            # GaussianDiffusion 调度器
│   ├── cnn.py, resnet.py           # 分类网络
│   ├── vae.py                      # ConvVAE
│   ├── sr.py                       # SimpleSRResNet
│   ├── sr3.py                      # SR3UNet（继承 UNetModel，改 in_channels=2）
│   └── resshift.py                 # ResShiftUNet + ResidualShiftScheduler
├── trainer/
│   ├── base.py                     # BaseTrainer 通用训练骨架
│   ├── vae.py                      # VAETrainer
│   ├── diffusion.py                # DiffusionTrainer
│   ├── sr.py                       # BaseSRTrainer + SuperResolutionTrainer
│   ├── sr3.py                      # SR3Trainer
│   └── resshift.py                 # ResShiftTrainer
├── utils/                          # 日志、指标、随机种子
├── main.py                         # 总入口：读配置 → 选 trainer → 启动训练
└── inference_vae.py                # VAE 推理脚本
```

> 想理解每层的职责和调用关系，看 [项目架构导读](docs/architecture.md)。

## 超分任务怎么跑

超分训练不直接复用分类标签，走一条单独的数据流：

1. 从 MNIST / FashionMNIST / CIFAR10 读取高分辨率图像
2. 下采样构造低分辨率图像
3. 把低分辨率图像插值回原尺寸
4. 返回 `(lr_up, hr)` 给训练器

不需要先准备 DIV2K 那种大数据集，用 MNIST 就能跑通整个流程。[data/sr_dataset.py](data/sr_dataset.py) 负责这个包装。

三种超分方法的区别：

| 方法 | 步数 | 预测目标 | 采样起点 | 配置 |
|------|------|----------|----------|------|
| SRResNet | 1 | 残差 | — | [srresnet.yaml](configs/sr/srresnet.yaml) |
| SR3 | 50-100 | 噪声 | 纯噪声 | [sr3.yaml](configs/sr/sr3.yaml) |
| ResShift | ~15 | 残差 | 低清图附近 | [resshift.yaml](configs/sr/resshift.yaml) |

三种方法的详细对比见 [生成模型知识补充 > ResShift](docs/generative_basics.md#resshift) 和 [ResShift 学习说明](docs/resshift.md)。

加速实验：

```yaml
data:
  max_train_samples: 1024
  max_val_samples: 256
```

## 输出内容

- 分类任务：保存 `last.pth / best.pth`
- VAE / DDPM：保存权重与生成样例
- SR / SR3 / ResShift：额外保存 `sr_epoch_x.png`（低清输入 / 模型输出 / 高分真值，三行对比）

## 推理脚本

训练完想单独做推理，有两个脚本可以用：

```bash
# VAE 推理：从潜空间采样生成图像
python inference_vae.py --ckpt checkpoints/vae_mnist/best.pth

# 超分推理：加载 checkpoint 生成对比图（支持 SRResNet / SR3 / ResShift）
python inference_sr.py --ckpt checkpoints/resshift_mnist_toy/best.pth
python inference_sr.py --ckpt checkpoints/sr3_mnist_toy/best.pth --device cpu
```

`inference_sr.py` 会输出低清输入 / 模型输出 / 高清真值的并排对比图，并打印 PSNR。

三种超分方法一键对比：

```bash
python compare_sr.py \
    --sr_ckpt checkpoints/srresnet_mnist/best.pth \
    --sr3_ckpt checkpoints/sr3_mnist_toy/best.pth \
    --resshift_ckpt checkpoints/resshift_mnist_toy/best.pth
```

生成五行并排对比图（低清 / SRResNet / SR3 / ResShift / 高清），并打印各自 PSNR。

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
  value_range: zero_one       # zero_one 或 minus_one_one
```

## 工程设计

这个项目的工程骨架值得单独看。[trainer/base.py](trainer/base.py) 不再只服务分类——每个任务声明自己监控什么指标（准确率 / loss / PSNR），基类统一负责比较 best 和保存 checkpoint。

新增任务只需要三步：写一个 trainer 类（继承 BaseTrainer）→ 在 TRAINER_REGISTRY 注册 → 写一个 yaml 配置。main.py 不用改。这就是注册表模式的好处。

> 详细的设计思路和扩展练习见 [项目架构导读](docs/architecture.md)。

## 论文与资料

| 资料 | 链接 |
|------|------|
| ResShift 论文 | <https://arxiv.org/abs/2307.12348> |
| ResShift 官方仓库 | <https://github.com/zsyOAOA/ResShift> |
| SR3 论文 | <https://arxiv.org/abs/2104.07636> |

## 依赖

```bash
pip install -r requirements.txt
```

`torch`、`torchvision`、`pyyaml`、`numpy`、`matplotlib`。
