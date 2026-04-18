# 项目架构导读

如果你第一次读这个仓库，不要先钻进某个模型文件里。这个项目真正想教会你的，不只是某一个网络怎么写，而是一个深度学习项目通常会分成哪几层，每一层分别承担什么职责。

## 一句话理解这个仓库

这是一个"配置驱动"的训练框架。

它用同一套训练入口，串起了六条学习主线：

- CNN / ResNet 分类
- VAE 生成
- DDPM 扩散
- 超分基线
- SR3 条件扩散超分
- ResShift 风格的条件扩散超分

换句话说，项目想让你看到的是：

同一套工程骨架，可以承载不同任务；变化的主要是数据定义、模型结构和训练目标。

## 配套知识文档

建议把这几份文档配合着看：

| 文档 | 讲什么 | 什么时候看 |
|------|--------|-----------|
| `docs/architecture.md` | 项目分层和调用链 | 第一次接触项目时 |
| `docs/learning_path.md` | 整体顺序和每个阶段应该看什么 | 规划学习路线时 |
| `docs/generative_basics.md` | VAE、DDPM、SR3、ResShift 的核心概念 | 学到对应模型时 |
| `docs/resshift.md` | 专门讲简化版 ResShift 的定位和思路 | 学到 ResShift 时 |
| `docs/mit_6s184_flow_matching_notes.md` | MIT 课程与仓库代码的对照 | 想深入理论时 |

---

## 先记住四层结构

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

### 1. `configs/`：实验说明书

配置文件决定这次实验要做什么。

比如一份配置通常会回答这些问题：

| 配置项 | 含义 | 示例 |
|--------|------|------|
| `train.epochs` | 训练多少轮 | `10` |
| `train.lr` | 学习率 | `0.001` |
| `train.batch_size` | 批大小 | `128` |
| `model.name` | 用哪个模型 | `cnn` / `vae` / `ddpm` |
| `trainer_name` | 用哪个训练器 | `base` / `vae` / `ddpm` |
| `data.dataset` | 数据集 | `mnist` / `cifar10` |
| `data.value_range` | 像素范围 | `zero_one` 或 `minus_one_one` |

你可以把配置理解成"给程序看的实验计划表"。改配置不需要改代码，这就是"配置驱动"的核心思想。

**配置目录结构**：

```
configs/
├── classification/       # 分类任务配置
│   ├── cnn.yaml          # CNN 手写数字识别
│   └── resnet.yaml       # ResNet 手写数字识别
├── generate/             # 生成任务配置
│   ├── vae.yaml          # VAE 图片生成
│   └── ddpm.yaml         # DDPM 扩散生成
└── sr/                   # 超分任务配置
    ├── srresnet.yaml     # SRResNet 超分基线
    ├── sr3.yaml          # SR3 条件扩散超分
    └── resshift.yaml     # ResShift 少步扩散超分
```

### 2. `main.py`：总入口

`main.py` 本身不做复杂训练，它主要负责：

```
读取配置 → 设置随机种子 → 创建 dataloader → 根据 trainer_name 找到对应 trainer → 调用 trainer.fit()
```

具体来说，核心代码只有这几步：

```python
# 1. 读取配置
with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 2. 设置随机种子
set_seed(config["seed"])

# 3. 创建 dataloader
train_loader, test_loader = get_dataloader(config)

# 4. 从注册表中查找 trainer
trainer_name = config.get("trainer_name", "base")
trainer_class = TRAINER_REGISTRY[trainer_name]

# 5. 实例化并启动训练
trainer = trainer_class(config=config, train_loader=train_loader, val_loader=test_loader)
trainer.fit()
```

这说明入口文件越薄越好。复杂逻辑如果都堆在入口里，后面一扩展任务就会很乱。

### 3. `trainer/`：训练流程层

trainer 是整个项目最关键的抽象层。

它负责的是"怎么训练"，而不是"网络长什么样"。

#### trainer 继承体系

```
BaseTrainer
├── VAETrainer           # 覆写 loss 和 evaluate
├── DiffusionTrainer     # 覆写 loss、增加采样可视化
└── BaseSRTrainer        # 覆写 evaluate（PSNR）、增加可视化
    ├── SuperResolutionTrainer  # 覆写 loss（L1）、train_one_epoch
    ├── SR3Trainer              # 覆写 loss（MSE噪声预测）、增加扩散调度器
    └── ResShiftTrainer         # 覆写 loss（MSE残差预测）、增加残差调度器
```

以 `BaseTrainer` 为例，它负责统一处理：

| 职责 | 方法 | 说明 |
|------|------|------|
| 设备选择 | `_build_device()` | 根据 config 选择 cuda / mps / cpu |
| 模型构建 | `_build_model()` | 从 `MODEL_REGISTRY` 中查找并实例化 |
| 优化器构建 | `_build_optimizer()` | 默认 Adam |
| checkpoint 保存 | `save_checkpoint()` | 保存 `last.pth` 和 `best.pth` |
| 训练循环 | `train_one_epoch()` | 默认实现分类训练 |
| 验证循环 | `evaluate()` | 默认实现分类验证 |
| 最优指标监控 | `_is_improvement()` | 根据 monitor_mode 判断是否改善 |
| 完整训练 | `fit()` | 串联训练、验证、保存 |

而具体任务只需要覆写少量 hook：

| 任务 | 覆写了什么 | 不变的是什么 |
|------|-----------|-------------|
| 分类（默认） | 无，直接用基类 | 全部 |
| VAE | `_build_criterion()`、`train_one_epoch()`、`evaluate()` | `fit()`、checkpoint 逻辑 |
| DDPM | `_build_criterion()`、`train_one_epoch()`、`evaluate()`，新增 `_save_samples()` | `fit()`、checkpoint 逻辑 |
| SR | `_build_criterion()`、`train_one_epoch()`、`infer()` | `evaluate()`（在 BaseSRTrainer 中统一实现） |
| SR3 | 同上，新增 `self.diffusion` | `evaluate()`、可视化逻辑 |
| ResShift | 同上，新增 `self.scheduler` | `evaluate()`、可视化逻辑 |

这里的设计重点是：

**把"通用流程"放在父类，把"任务差异"放在子类。**

#### 注册表模式

`trainer/__init__.py` 中维护了一个注册表：

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

`main.py` 通过 `config["trainer_name"]` 从注册表中查找对应的类，不需要写 `if-else` 分支。

**如果要新增一个任务**，只需要：
1. 写一个新的 trainer 类，继承 `BaseTrainer` 或其子类
2. 在 `TRAINER_REGISTRY` 中注册
3. 写一个新的 yaml 配置文件

不需要修改 `main.py` 的任何代码。

### 4. `models/`：网络定义层

model 的职责应该尽量单纯：

- 接受输入
- 前向传播
- 输出预测结果

model **不应该**知道：

- checkpoint 怎么保存
- dataloader 怎么构造
- 实验目录叫什么
- 这次训练的最佳指标是多少

这类事情应该归 trainer 管。

#### models 目录结构

```
models/
├── __init__.py          # MODEL_REGISTRY 注册表
├── cnn.py               # SimpleCNN 分类网络
├── resnet.py            # ResNet 分类网络
├── vae.py               # ConvVAE 变分自编码器
├── sr.py                # SimpleSRResNet 超分网络
├── sr3.py               # SR3UNet 条件扩散 UNet（继承 UNetModel）
├── resshift.py          # ResShiftUNet + ResidualShiftScheduler
└── ddpm/
    ├── unet.py          # UNetModel 扩散去噪网络
    └── diffusion.py     # GaussianDiffusion 扩散调度器
```

#### 一个关键观察：SR3 和 ResShift 的 UNet

`SR3UNet` 和 `ResShiftUNet` 都继承自 `UNetModel`，唯一的区别是 `in_channels`：

```python
# models/sr3.py
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):  # 2通道 = 噪声图 + 条件图
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)

# models/resshift.py
class ResShiftUNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):  # 2通道 = 中间态 + 条件图
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

这说明"条件扩散"和"无条件扩散"在网络结构上的差异其实很小——关键变化在于训练逻辑和调度器，而不是网络本身。

---

## 一次训练到底是怎么跑起来的

推荐你按下面顺序跟代码，把整条调用链串起来：

### 以 CNN 分类为例的完整调用链

```
1. configs/classification/cnn.yaml
   ↓ 定义 trainer_name: base, model.name: cnn, data.dataset: mnist

2. main.py
   ↓ 读取配置 → set_seed() → get_dataloader() → TRAINER_REGISTRY["base"] → BaseTrainer(config, ...)

3. trainer/base.py → __init__()
   ↓ _build_device() → _build_model() → _build_criterion() → _build_optimizer()
   ↓ 创建实验目录、保存配置

4. trainer/base.py → fit()
   ↓ 循环 epochs:
   ↓   train_one_epoch(epoch)  → 前向 → loss → 反向 → 更新
   ↓   evaluate(epoch)         → 计算验证准确率
   ↓   _is_improvement()       → 判断是否是 best
   ↓   save_checkpoint()       → 保存 last.pth / best.pth

5. models/cnn.py → forward(x)
   ↓ x: (B,1,28,28) → features → classifier → (B,10)
```

### 以 ResShift 超分为例的完整调用链

```
1. configs/sr/resshift.yaml
   ↓ 定义 trainer_name: resshift, model.name: resshift, data.task: super_resolution

2. main.py
   ↓ 读取配置 → get_dataloader() → _infer_task() → "super_resolution"
   ↓ → SyntheticSuperResolutionDataset 包装数据 → (lr_up, hr) 训练对

3. trainer/resshift.py → __init__()
   ↓ 创建 ResidualShiftScheduler(timesteps=15, noise_level=0.15)
   ↓ → super().__init__() → BaseSRTrainer → BaseTrainer

4. trainer/resshift.py → train_one_epoch(epoch)
   ↓ for lr, hr in train_loader:
   ↓   t = randint(0, timesteps)
   ↓   shifted, residual = scheduler.q_sample(hr, lr, t)  # 构造中间态
   ↓   predicted_residual = model(cat([shifted, lr], dim=1), t)
   ↓   loss = MSE(predicted_residual, residual)
   ↓   backward + step

5. trainer/sr.py → BaseSRTrainer.evaluate(epoch)
   ↓ for lr, hr in val_loader:
   ↓   sr = infer(lr)  → scheduler.sample(model, lr)  # 15步逐步恢复
   ↓   psnr = calculate_psnr(sr, hr)
   ↓ _save_visuals(lr, sr, hr)
```

### 跟代码的建议

1. 打开 `configs/...yaml`，看清楚这次实验的配置
2. 看 `main.py`，理解配置如何被读取和使用
3. 看 `trainer/__init__.py`，找到对应的 trainer 类
4. 看具体的 trainer 文件，理解训练和验证逻辑
5. 再去看对应 model，理解网络结构
6. 最后回来看 `data/` 怎么准备样本

把这条链路串起来，你就能真正理解项目，而不是只会"跑命令"。

---

## 数据层在教什么

这个项目的数据层刻意保持得比较直观。

### 数据层文件结构

```
data/
├── dataloader.py    # 统一入口：根据配置构建 DataLoader
├── sr_dataset.py    # 超分数据包装：把 (image, label) 变成 (lr_up, hr)
└── transforms.py    # 图像预处理：Resize → ToTensor → 值域映射
```

### 标准任务

分类、VAE、DDPM 这些任务默认还是读取普通图像数据集，比如 `MNIST`。

原始样本长这样：

- 分类：`(image, label)`，label 是 0-9 的整数
- 生成：`(image, label)`，虽然 label 通常不用（VAE 和 DDPM 的 trainer 里用 `_` 忽略了 label）

### 超分任务

超分不是直接拿分类数据来训，而是先把原始图像包装成新的训练对。

`data/sr_dataset.py` 中的 `SyntheticSuperResolutionDataset` 做了这些事：

```
1. 从原始数据集中拿到高分辨率图 hr
2. 对它做下采样，得到低分图 lr          ← F.interpolate(hr, size=lr_size, mode="bicubic")
3. 再把 lr 插值回原尺寸，得到 lr_up     ← F.interpolate(lr, size=hr_size, mode="bicubic")
4. 可选：给 lr 加少量噪声               ← lr + noise * noise_std
5. 返回 (lr_up, hr)
```

这里最重要的工程思想是：

**原始数据集和任务样本不是一回事。**

同一个基础数据集，可以被包装成分类任务、生成任务、超分任务等不同训练样本。`data/dataloader.py` 中的 `_infer_task()` 函数会根据 `trainer_name` 自动判断任务类型，并决定是否需要用 `SyntheticSuperResolutionDataset` 包装。

### 值域处理

`data/transforms.py` 负责图像预处理，其中 `value_range` 的选择很重要：

| 值域 | 范围 | 适用任务 | 原因 |
|------|------|----------|------|
| `zero_one` | [0, 1] | 分类、VAE、SRResNet、ResShift | BCE 需要 [0,1] 输入；L1/L2 在 [0,1] 上更稳定 |
| `minus_one_one` | [-1, 1] | DDPM、SR3 | 扩散模型加噪后值域对称，训练更稳定 |

转换公式在 `transforms.py` 中：`x * 2.0 - 1.0`

---

## 五条主线各自教什么

### 1. CNN / ResNet

这是最基础的一条线，用来理解：

| 学什么 | 代码位置 | 关键点 |
|--------|----------|--------|
| 配置驱动训练入口 | `configs/classification/cnn.yaml` → `main.py` | 改配置不改代码 |
| 标准监督学习循环 | `trainer/base.py` 的 `train_one_epoch()` | 前向 → loss → 反向 → 更新 |
| 分类指标计算 | `trainer/base.py` 的 `evaluate()` | `logits.argmax(dim=1)` 得到预测类别 |

### 2. VAE

这条线帮助你理解：

| 学什么 | 代码位置 | 关键点 |
|--------|----------|--------|
| 模型输出不只是一个张量 | `models/vae.py` 的 `forward()` | 返回 `(x_recon, mu, logvar)` 三元组 |
| loss 由多项组成 | `trainer/vae.py` 的 `_build_criterion()` | 重构损失 + KL 损失 |
| 验证指标越小越好 | `trainer/vae.py` 的 `_monitor_mode()` | 返回 `"min"` 而非 `"max"` |
| 潜空间和生成 | `models/vae.py` 的 `reparameterize()` | 重参数化让采样可导 |

### 3. DDPM

这里开始引入"时间步"。

| 学什么 | 代码位置 | 关键点 |
|--------|----------|--------|
| 时间步嵌入 | `models/ddpm/unet.py` 的 `timestep_embedding()` | 把整数 t 编码成向量 |
| 加噪过程 | `models/ddpm/diffusion.py` 的 `q_sample()` | `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` |
| 去噪过程 | `models/ddpm/diffusion.py` 的 `p_sample()` | 从预测噪声推算 x_{t-1} |
| 训练目标变了 | `trainer/diffusion.py` | 预测噪声，不是预测类别 |
| 采样过程是设计的一部分 | `trainer/diffusion.py` 的 `_save_samples()` | T 步迭代去噪 |

### 4. SR 基线

这条线帮助你理解"图像恢复"和"图像分类"的差别。

| 学什么 | 代码位置 | 关键点 |
|--------|----------|--------|
| 数据包装 | `data/sr_dataset.py` | `(image, label)` → `(lr_up, hr)` |
| 残差学习 | `models/sr.py` | `return x + residual` |
| PSNR 指标 | `utils/metrics.py` | `10 * log10(data_range^2 / MSE)` |
| 超分可视化 | `trainer/sr.py` 的 `_save_visuals()` | 低清/超分/高清三行对比 |

### 5. ResShift

这是本项目最适合继续往扩散恢复方向延伸的一条扩展线。

| 学什么 | 代码位置 | 关键点 |
|--------|----------|--------|
| 残差调度器 | `models/resshift.py` 的 `ResidualShiftScheduler` | `residual_scales` 从 1.0 降到 0.0 |
| 训练中间态构造 | `ResidualShiftScheduler.q_sample()` | `shifted = condition + scale * residual + noise_scale * noise` |
| 少步采样 | `ResidualShiftScheduler.sample()` | 只需 15 步 |
| 残差预测 | `trainer/resshift.py` | `loss = MSE(predicted_residual, residual)` |

你可以把它理解成：

- SRResNet：一步直接修图
- ResShift：带时间步、少步数地逐步恢复残差

在它之前，先看 SR3，会更容易理解"条件扩散超分"。

如果你对这些概念还不熟，建议直接接着看 `docs/generative_basics.md` 和 `docs/learning_path.md`。

---

## 这次结构更新解决了什么问题

### 问题 1：不同任务的"最佳指标"逻辑分散

以前分类看 `Acc`，VAE 看 `Loss`，SR 看 `PSNR`，每个 trainer 都各写一套 best/save 逻辑。

这会带来两个问题：

- 重复代码多
- 容易出现某个任务保存逻辑和基类不一致

现在统一改成了"监控指标"机制：

```python
# 每个 trainer 只需声明自己监控什么指标
def _monitor_name(self):       return "val_acc"       # 或 "val_loss" / "val_psnr"
def _monitor_display_name(self): return "Val Acc"      # 或 "Val Loss" / "Val PSNR"
def _monitor_mode(self):       return "max"           # 或 "min"
def _monitor_unit(self):       return ""              # 或 "dB"

# 基类统一负责比较、保存 best、保存 last
def _is_improvement(self, metric):
    if self.monitor_mode == "max": return metric > self.best_metric
    return metric < self.best_metric
```

这也是实际工程里常见的重构方向。

### 问题 2：DDPM 的 checkpoint 语义不清楚

以前 DDPM 为了兼容分类 trainer，会返回一个假的指标值去触发保存。

现在 DDPM 会：

- 训练时记录噪声预测损失
- 验证时计算 `Val Noise Loss`
- 同时额外保存采样图片（`_save_samples()`）

这样"指标"和"可视化"各司其职，更容易理解。

### 问题 3：数据层文件职责太重

以前 `data/dataloader.py` 同时负责：

- transform
- SR 包装数据集
- dataset registry
- dataloader 构造

现在把最容易讲解的两部分拆了出来：

- `data/transforms.py`：只负责图像预处理
- `data/sr_dataset.py`：只负责超分数据包装

这样看数据层时，不会一上来就被一个大文件淹没。

---

## 建议的阅读顺序

推荐这样看：

| 步骤 | 文件 | 你应该理解什么 |
|------|------|---------------|
| 1 | `configs/classification/cnn.yaml` | 配置长什么样，每个字段控制什么 |
| 2 | `main.py` | 配置如何被读取，trainer 如何被选择 |
| 3 | `trainer/base.py` | 通用训练骨架：构建、训练、验证、保存 |
| 4 | `models/cnn.py` | 一个简单的 CNN 长什么样 |
| 5 | `trainer/vae.py` | 如何覆写基类方法实现不同训练逻辑 |
| 6 | `trainer/diffusion.py` | 如何引入时间步和噪声预测 |
| 7 | `data/sr_dataset.py` | 如何把分类数据包装成超分数据 |
| 8 | `trainer/sr.py` | 超分训练器的公共基类 |
| 9 | `trainer/resshift.py` | 如何在框架内实现新的算法 |
| 10 | `models/resshift.py` | 调度器和网络如何配合 |

顺序不要反过来。先看流程，再看模型，效率会高很多。

---

## 可以继续做的扩展练习

如果你已经读懂这套结构，可以尝试下面几个练习：

1. **给分类任务增加 `FashionMNIST` 配置**：
   - 复制 `cnn.yaml`，改 `data.dataset: fashion_mnist`
   - 不需要改任何代码

2. **给 `BaseTrainer` 增加学习率调度器支持**：
   - 新增 `_build_scheduler()` 方法
   - 在 `fit()` 的 epoch 循环中调用 `scheduler.step()`

3. **给超分任务增加 `SSIM` 指标**：
   - 在 `utils/metrics.py` 中新增 `calculate_ssim()` 函数
   - 在 `BaseSRTrainer.evaluate()` 中同时计算 PSNR 和 SSIM

4. **把 DDPM 的采样图保存频率做成可配置项**：
   - 在 yaml 中增加 `diffusion.sample_interval: 5`
   - 在 `DiffusionTrainer.evaluate()` 中判断是否需要保存

5. **增加一个新的 trainer，比如去噪任务**：
   - 继承 `BaseTrainer`
   - 覆写 `_build_criterion()`、`train_one_epoch()`、`evaluate()`
   - 在 `TRAINER_REGISTRY` 中注册
   - 写一个新的 yaml 配置

真正学会工程，不是把现有代码跑通，而是你能在现有结构上继续扩展。

---

## 各组件职责速查表

| 组件 | 职责 | 不应该做的事 |
|------|------|-------------|
| `configs/*.yaml` | 定义实验参数 | 不包含任何逻辑 |
| `main.py` | 读配置、选组件、启动训练 | 不包含训练逻辑 |
| `trainer/` | 训练流程、loss 定义、指标计算 | 不定义网络结构 |
| `models/` | 网络结构、前向传播 | 不关心训练流程 |
| `data/` | 数据加载、预处理、包装 | 不关心模型和训练 |
| `utils/` | 通用工具（日志、指标、种子） | 不依赖具体任务 |
