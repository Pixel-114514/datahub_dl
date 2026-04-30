# 项目架构导读

第一次看这个仓库，别急着钻进某个模型文件。这个项目想教你的不只是某一个网络怎么写，而是一个深度学习项目通常怎么分层，每层负责什么。

## 一句话理解

这是一个配置驱动的训练框架。同一套 `main.py` 入口，通过换配置文件就能跑六种不同任务：CNN/ResNet 分类、VAE 生成、DDPM 扩散、超分基线、SR3 条件扩散超分、ResShift 少步扩散超分。

变化的主要是数据定义、模型结构和训练目标。工程骨架是同一套。

配合着看的文档：

| 文档 | 讲什么 |
|------|--------|
| `docs/architecture.md` | 项目分层和调用链（就是这份） |
| `docs/learning_path.md` | 整体学习顺序 |
| `docs/generative_basics.md` | VAE、DDPM、SR3、ResShift 的核心概念 |
| `docs/resshift.md` | 简化版 ResShift 的定位和思路 |
| `docs/mit_6s184_flow_matching_notes.md` | MIT 课程与仓库代码的对照 |

---

## 四层结构

```
┌─────────────────────────────────────────────────┐
│  configs/*.yaml   ← 实验说明书（决定做什么）       │
├─────────────────────────────────────────────────┤
│  main.py          ← 总入口（读配置、选trainer、启动）│
├─────────────────────────────────────────────────┤
│  trainer/         ← 训练流程层（决定怎么训）        │
├─────────────────────────────────────────────────┤
│  models/          ← 网络定义层（决定网络长什么样）   │
└─────────────────────────────────────────────────┘
         ↕
┌─────────────────────────────────────────────────┐
│  data/            ← 数据层（决定喂什么数据）        │
└─────────────────────────────────────────────────┘
```

### configs/：实验说明书

配置文件回答"这次实验做什么"：

```yaml
train:
  epochs: 10
  lr: 0.001
  batch_size: 128
model:
  name: cnn
trainer_name: base
data:
  dataset: mnist
```

改配置不用改代码，这就是配置驱动的核心。配置目录结构：

```
configs/
├── classification/       # cnn.yaml, resnet.yaml
├── generate/             # vae.yaml, ddpm.yaml
└── sr/                   # srresnet.yaml, sr3.yaml, resshift.yaml
```

### main.py：总入口

`main.py` 本身不做复杂训练，只做三件事：

```python
# 1. 读配置
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 设随机种子、创建 dataloader
set_seed(config["seed"])
train_loader, test_loader = get_dataloader(config)

# 3. 从注册表找 trainer，实例化并启动
trainer_class = TRAINER_REGISTRY[config.get("trainer_name", "base")]
trainer = trainer_class(config=config, train_loader=train_loader, val_loader=test_loader)
trainer.fit()
```

入口文件越薄越好。复杂逻辑堆在入口里，后面一扩展就乱。

### trainer/：训练流程层

trainer 是整个项目最关键的抽象。它负责"怎么训练"，不负责"网络长什么样"。

继承体系：

```
BaseTrainer
├── VAETrainer           # 覆写 loss 和 evaluate
├── DiffusionTrainer     # 覆写 loss、增加采样可视化
└── BaseSRTrainer        # 覆写 evaluate（PSNR）、增加可视化
    ├── SuperResolutionTrainer  # 覆写 loss（L1）
    ├── SR3Trainer              # 覆写 loss（MSE噪声预测）、增加扩散调度器
    └── ResShiftTrainer         # 覆写 loss（MSE残差预测）、增加残差调度器
```

`BaseTrainer` 统一处理设备选择、模型构建、优化器构建、checkpoint 保存、训练循环、验证循环、最优指标监控。具体任务只需覆写少量方法。

| 任务 | 覆写了什么 | 不变的是什么 |
|------|-----------|-------------|
| 分类（默认） | 无 | 全部 |
| VAE | loss、train_one_epoch、evaluate | fit()、checkpoint |
| DDPM | loss、train_one_epoch、evaluate，新增采样保存 | fit()、checkpoint |
| SR / SR3 / ResShift | loss、train_one_epoch、infer | evaluate（BaseSRTrainer 统一实现） |

设计原则：**通用流程放父类，任务差异放子类。**

#### 注册表模式

`trainer/__init__.py` 里维护一个注册表：

```python
TRAINER_REGISTRY = {
    "base": BaseTrainer,
    "vae": VAETrainer,
    "ddpm": DiffusionTrainer,
    "sr": SuperResolutionTrainer,
    "sr3": SR3Trainer,
    "resshift": ResShiftTrainer,
}
```

`main.py` 通过 `config["trainer_name"]` 查找，不需要写 if-else。新增任务只需要：写一个 trainer 类 → 注册 → 写 yaml 配置。`main.py` 不用改。

#### 监控指标机制

不同任务看不同指标：分类看准确率（越大越好），VAE 看 loss（越小越好），超分看 PSNR（越大越好）。

以前每个 trainer 各写一套 best/save 逻辑，容易不一致。现在统一了：

```python
# 每个 trainer 声明自己监控什么
def _monitor_name(self):        return "val_acc"    # 或 "val_loss" / "val_psnr"
def _monitor_display_name(self): return "Val Acc"
def _monitor_mode(self):        return "max"        # 或 "min"

# 基类统一比较、保存
def _is_improvement(self, metric):
    if self.monitor_mode == "max": return metric > self.best_metric
    return metric < self.best_metric
```

### models/：网络定义层

model 的职责应该尽量单纯：接受输入、前向传播、输出结果。model 不应该知道 checkpoint 怎么保存、dataloader 怎么构造、实验目录叫什么。

```
models/
├── __init__.py          # MODEL_REGISTRY
├── cnn.py               # SimpleCNN
├── resnet.py            # ResNet
├── vae.py               # ConvVAE
├── sr.py                # SimpleSRResNet
├── sr3.py               # SR3UNet（继承 UNetModel）
├── resshift.py          # ResShiftUNet + ResidualShiftScheduler
└── ddpm/
    ├── unet.py          # UNetModel
    └── diffusion.py     # GaussianDiffusion
```

一个值得注意的细节：`SR3UNet` 和 `ResShiftUNet` 都继承 `UNetModel`，唯一区别是 `in_channels`：

```python
# models/sr3.py
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

条件扩散和无条件扩散在网络结构上差异很小——关键变化在训练逻辑和调度器。

---

## 一次训练怎么跑起来的

以 CNN 分类为例跟一遍调用链：

```
configs/classification/cnn.yaml
  ↓ 定义 trainer_name: base, model.name: cnn, data.dataset: mnist

main.py
  ↓ 读配置 → set_seed() → get_dataloader() → TRAINER_REGISTRY["base"]

trainer/base.py → __init__()
  ↓ _build_device() → _build_model() → _build_criterion() → _build_optimizer()

trainer/base.py → fit()
  ↓ 循环 epochs:
  ↓   train_one_epoch() → 前向 → loss → 反向 → 更新
  ↓   evaluate()         → 验证准确率
  ↓   _is_improvement()  → 判断是否 best
  ↓   save_checkpoint()  → 保存 last.pth / best.pth

models/cnn.py → forward(x)
  ↓ (B,1,28,28) → features → classifier → (B,10)
```

以 ResShift 超分为例：

```
configs/sr/resshift.yaml
  ↓ trainer_name: resshift, model.name: resshift, data.task: super_resolution

main.py
  ↓ get_dataloader() → _infer_task() → "super_resolution"
  ↓ → SyntheticSuperResolutionDataset 包装数据 → (lr_up, hr)

trainer/resshift.py → __init__()
  ↓ 创建 ResidualShiftScheduler(timesteps=15, noise_level=0.15)

trainer/resshift.py → train_one_epoch()
  ↓ for lr, hr in train_loader:
  ↓   t = randint(0, timesteps)
  ↓   shifted, residual = scheduler.q_sample(hr, lr, t)
  ↓   predicted_residual = model(cat([shifted, lr], dim=1), t)
  ↓   loss = MSE(predicted_residual, residual)

trainer/sr.py → evaluate()
  ↓ sr = scheduler.sample(model, lr)  # 15步恢复
  ↓ psnr = calculate_psnr(sr, hr)
```

跟代码的建议顺序：先看 yaml 配置 → `main.py` → `trainer/__init__.py` 找到对应类 → 具体 trainer 文件 → 再看 model → 最后看 data。

---

## 数据层

数据层文件结构：

```
data/
├── dataloader.py    # 统一入口：根据配置构建 DataLoader
├── sr_dataset.py    # 超分数据包装：(image, label) → (lr_up, hr)
└── transforms.py    # 图像预处理：Resize → ToTensor → 值域映射
```

分类、VAE、DDPM 直接读普通数据集（如 MNIST），返回 `(image, label)`。VAE 和 DDPM 的 trainer 里用 `_` 忽略 label。

超分数据要额外包装一下。`SyntheticSuperResolutionDataset` 做的事：

1. 拿到高分辨率图 `hr`
2. 下采样得到 `lr`（`F.interpolate(hr, size=lr_size, mode="bicubic")`）
3. 插值回原尺寸得到 `lr_up`
4. 可选加少量噪声
5. 返回 `(lr_up, hr)`

`_infer_task()` 根据 `trainer_name` 自动判断任务类型，决定是否需要包装。

值域选择：

| 值域 | 范围 | 适用任务 |
|------|------|----------|
| `zero_one` | [0, 1] | 分类、VAE、SRResNet、ResShift |
| `minus_one_one` | [-1, 1] | DDPM、SR3 |

扩散模型用 [-1,1] 是因为加噪后值域对称，训练更稳定。转换公式：`x * 2.0 - 1.0`。

---

## 这次结构更新解决了什么

### DDPM 的 checkpoint 语义

以前 DDPM 为了兼容分类 trainer，返回一个假指标值去触发保存。现在 DDPM 训练时记录噪声预测损失，验证时计算 `Val Noise Loss`，同时额外保存采样图片。指标和可视化各司其职。

### 数据层职责太重

以前 `dataloader.py` 同时负责 transform、SR 包装、dataset registry、dataloader 构造。现在把 `transforms.py` 和 `sr_dataset.py` 拆出来单独看，不会一上来被大文件淹没。

---

## 建议的阅读顺序

| 步骤 | 文件 | 看什么 |
|------|------|--------|
| 1 | `configs/classification/cnn.yaml` | 配置长什么样 |
| 2 | `main.py` | 配置怎么被读取、trainer 怎么被选择 |
| 3 | `trainer/base.py` | 通用训练骨架 |
| 4 | `models/cnn.py` | 一个简单 CNN |
| 5 | `trainer/vae.py` | 如何覆写基类方法 |
| 6 | `trainer/diffusion.py` | 如何引入时间步和噪声预测 |
| 7 | `data/sr_dataset.py` | 如何把分类数据包装成超分数据 |
| 8 | `trainer/sr.py` | 超分训练器的公共基类 |
| 9 | `trainer/resshift.py` | 如何在框架内实现新算法 |
| 10 | `models/resshift.py` | 调度器和网络怎么配合 |

顺序不要反过来。先看流程，再看模型，效率高很多。

---

## 扩展练习

读懂结构后可以试：

1. **给分类加 FashionMNIST 配置**：复制 `cnn.yaml`，改 `data.dataset: fashion_mnist`，不改代码。

2. **给 BaseTrainer 加学习率调度器**：新增 `_build_scheduler()`，在 `fit()` 的 epoch 循环中调 `scheduler.step()`。

3. **给超分加 SSIM 指标**：在 `utils/metrics.py` 新增 `calculate_ssim()`，在 `BaseSRTrainer.evaluate()` 中同时算 PSNR 和 SSIM。

4. **把 DDPM 采样保存频率做成可配置**：yaml 中加 `diffusion.sample_interval: 5`，`evaluate()` 中判断是否需要保存。

5. **新增一个 trainer**：比如去噪任务——继承 `BaseTrainer`，覆写 loss/训练/验证，注册，写 yaml。

---

## 各组件职责速查

| 组件 | 职责 | 不应该做的事 |
|------|------|-------------|
| `configs/*.yaml` | 定义实验参数 | 不包含逻辑 |
| `main.py` | 读配置、选组件、启动训练 | 不包含训练逻辑 |
| `trainer/` | 训练流程、loss、指标 | 不定义网络结构 |
| `models/` | 网络结构、前向传播 | 不关心训练流程 |
| `data/` | 数据加载、预处理、包装 | 不关心模型和训练 |
| `utils/` | 通用工具（日志、指标、种子） | 不依赖具体任务 |
