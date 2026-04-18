# ResShift 学习说明

## 先说结论

ResShift 适合放进这条路径里，不是因为它"更复杂"，而是因为它把扩散式图像恢复里最关键的一个工程问题说得很直接：

- 传统扩散恢复模型效果强，但推理步数通常很多
- 对超分、去噪、去 JPEG 等恢复任务来说，输入本来就已经带有大量结构信息
- 因此没必要像无条件生成那样，从纯高斯噪声开始慢慢采样

ResShift 的核心思路，就是围绕"退化图像到目标图像之间的残差"去设计一个更短、更贴近恢复任务本质的迁移过程。

---

## 官方方法的核心点

参考论文和官方仓库，ResShift 主要强调下面几件事：

- 它把图像恢复看成一个 residual shifting 过程，而不是标准的从纯噪声生成图像
- 条件输入是退化图像，模型关注的是如何逐步把目标残差恢复出来
- 相比传统扩散恢复方法，它希望用更少的采样步数完成推理
- 官方项目把它用于 super-resolution、deblurring、de-noising、face restoration 等多类恢复任务

论文和官方仓库都强调了它的高效推理特性，常见设置会把采样步数控制在 15 步这一量级。

参考：

- 论文：<https://arxiv.org/abs/2307.12348>
- 官方仓库：<https://github.com/zsyOAOA/ResShift>

---

## 本仓库里的实现是什么

这里实现的是一个简化版 `ResShift`，目标不是复现论文全部细节，而是把下面这条链路讲清楚：

1. 先构造低清图 `LR_up`
2. 计算高分图和低清图之间的残差 `R = HR - LR_up`
3. 在不同时间步里，对这个残差施加不同强度的保留和噪声扰动
4. 让网络学习：给定当前状态 `x_t` 和条件图 `LR_up`，预测残差 `R`
5. 推理时用少量 step 逐步恢复残差，再重建高分结果

对应代码位置：

| 文件 | 职责 |
|------|------|
| `models/resshift.py` | `ResShiftUNet`（网络）+ `ResidualShiftScheduler`（调度器） |
| `trainer/resshift.py` | 训练逻辑：构造中间态 → 预测残差 → 计算 loss |
| `configs/sr/resshift.yaml` | 实验配置：timesteps、noise_level、schedule |

如果你还没有掌握 VAE、DDPM 和 SR3 的基本概念，建议先看 `docs/generative_basics.md`，再回来看这份文档，会更容易理解 ResShift 为什么是"条件恢复版的少步扩散思路"。

---

## ResShift 算法详解

### 核心思想：残差迁移

ResShift 的核心公式非常简洁：

```
shifted = condition + residual_scale[t] * residual + noise_scale[t] * noise
```

其中：

- `condition`：低清条件图 `LR_up`（固定不变）
- `residual`：`HR - LR_up`（目标残差）
- `residual_scale[t]`：时间步 `t` 处的残差保留比例，从 1.0 降到 0.0
- `noise_scale[t]`：时间步 `t` 处的噪声强度，随残差减少而增大

这个公式的直觉是：

- **t=0 时**：`shifted ≈ LR_up + 1.0 * residual + 0 = HR`，中间态接近目标图
- **t=T 时**：`shifted ≈ LR_up + 0 * residual + noise = LR_up + noise`，中间态接近低清图加噪声

所以 ResShift 的"扩散过程"不是从纯噪声到图像，而是从"低清图加噪声"到"高清图"。

### 调度器的构造

`ResidualShiftScheduler` 在初始化时预计算了两个关键数组：

```python
# models/resshift.py
if schedule == "linear":
    residual_scales = torch.linspace(1.0, 0.0, timesteps)  # 线性衰减
elif schedule == "cosine":
    residual_scales = torch.cos(torch.linspace(0.0, pi/2, timesteps))  # 余弦衰减

noise_scales = (1.0 - residual_scales) * noise_level
```

两种 schedule 的区别：

| Schedule | residual_scales 变化 | 特点 |
|----------|---------------------|------|
| `linear` | 从 1.0 匀速降到 0.0 | 简单直接，每步变化量相同 |
| `cosine` | 从 1.0 余弦降到 0.0 | 前期慢后期快，更平滑的过渡 |

**打个比方**：linear 就像匀速刹车，cosine 就像渐进刹车——开始轻踩，后面越来越重。

### 训练过程代码走读

`trainer/resshift.py` 中的 `train_one_epoch()` 是训练的核心：

```python
def train_one_epoch(self, epoch):
    self.model.train()
    for lr, hr in self.train_loader:
        lr, hr = lr.to(self.device), hr.to(self.device)

        # 1. 随机采样时间步
        batch_size = lr.size(0)
        t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device)

        # 2. 构造训练中间态和目标残差
        shifted, residual = self.scheduler.q_sample(hr, lr, t)
        # shifted: condition + scale * residual + noise_scale * noise
        # residual: HR - LR_up

        # 3. 模型预测残差
        predicted_residual = self.model(torch.cat([shifted, lr], dim=1), t)

        # 4. 计算残差预测 loss
        loss = self.criterion(predicted_residual, residual)  # MSE

        # 5. 反向传播和参数更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**关键观察**：

- 模型输入是 `torch.cat([shifted, lr], dim=1)`，即中间态和条件图的通道拼接
- 训练目标是残差 `residual = HR - LR_up`，不是噪声
- 每个样本的时间步 `t` 是随机采样的，模型需要学会在所有时间步上预测残差

### 推理过程代码走读

`models/resshift.py` 中的 `ResidualShiftScheduler.sample()` 是推理的核心：

```python
@torch.no_grad()
def sample(self, model, condition, clamp_range=None):
    batch_size = condition.shape[0]
    device = condition.device

    # 1. 从低清图附近开始（不是纯噪声！）
    noise_scale = self.noise_scales[-1].to(device)
    current = condition + noise_scale * torch.randn_like(condition)

    # 2. 逐步恢复
    for step in reversed(range(self.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)

        # 3. 模型预测残差
        predicted_residual = model(torch.cat([current, condition], dim=1), t)

        # 4. 最后一步直接用预测残差重建
        if step == 0:
            current = condition + predicted_residual
            continue

        # 5. 中间步骤：按调度器缩放残差并加噪声
        prev_t = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
        residual_scale = self._extract(self.residual_scales, prev_t, current.shape)
        noise_scale = self._extract(self.noise_scales, prev_t, current.shape)
        current = condition + residual_scale * predicted_residual
        current = current + noise_scale * torch.randn_like(current)

    # 6. 可选：裁剪到合法值域
    if clamp_range is not None:
        current = current.clamp(*clamp_range)
    return current
```

**关键观察**：

- 采样起点是 `condition + noise`，不是纯随机噪声
- 每一步都把 `condition`（低清图）作为锚点，预测的残差叠加在低清图上
- 最后一步（`step=0`）不加噪声，直接用预测残差重建
- 整个过程只需要 15 步（而非 DDPM 的 500 步）

---

## ResShift 和其他模型的关系

这些生成模型很容易混在一起，所以这里单独拆开说明。

### ResShift 和 VAE 的区别

VAE 的主线是：

- 学一个潜空间
- 从潜变量中采样
- 再解码生成图像

ResShift 不走潜空间重建这条路。

它更像是在像素空间里，围绕条件输入图像做一个带时间步的恢复过程。

| | VAE | ResShift |
|---|-----|----------|
| 建模空间 | 潜空间 | 像素空间 |
| 关键操作 | 编码 → 采样 → 解码 | 条件图 + 残差恢复 |
| 采样方式 | 一步解码 | 多步迭代 |
| 关键词 | 潜变量、KL 散度 | 条件图、残差、时间步恢复 |

### ResShift 和 DDPM 的区别

两者都用到了时间步和逐步恢复的思想，但它们解决的问题不同。

| | DDPM | ResShift |
|---|------|----------|
| 建模对象 | 噪声 ε | 残差 R = HR - LR |
| 采样起点 | 纯随机噪声 | 低清图 + 少量噪声 |
| 采样步数 | 500-1000 | ~15 |
| 条件信息 | 无 | 低清图 |
| 训练目标 | 预测噪声 | 预测残差 |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |
| 加噪公式 | `x_t = √ᾱ_t·x_0 + √(1-ᾱ_t)·ε` | `shifted = LR + scale·R + noise·ε` |
| 适用场景 | 无条件生成 | 条件图像恢复 |

**核心区别**：DDPM 从纯噪声"构建"图像，ResShift 从低清图"修补"图像。

### ResShift 和 SR3 的区别

如果前面已经看过 SR3，通常会问：

- 既然条件扩散已经能做超分，为什么还要学 ResShift

| | SR3 | ResShift |
|---|-----|----------|
| 训练目标 | 预测噪声 | 预测残差 |
| 采样起点 | 纯随机噪声 | 低清图附近 |
| 采样步数 | 50-100 | ~15 |
| 中间态构造 | 标准加噪 | 残差缩放 |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |
| 代码复用 | 复用 DDPM 的 diffusion.py | 自定义 scheduler |

这里最值得强调的是：

- SR3 解决"扩散怎么做超分"→ 证明了条件扩散超分可行
- ResShift 解决"扩散超分怎么做得更贴近恢复任务、更高效"→ 围绕残差设计更短的过程

### ResShift 和 SRResNet 的区别

| | SRResNet | ResShift |
|---|----------|----------|
| 采样步数 | 1 步 | ~15 步 |
| 输出方式 | 直接输出超分图 | 逐步恢复残差 |
| 时间步 | 无 | 有 |
| 表达能力 | 受限于单次前向传播 | 多步迭代，更强 |
| 推理速度 | 快 | 稍慢但远快于 DDPM |
| 代码 | `models/sr.py` | `models/resshift.py` |

**一句话总结**：

- SRResNet 是"一步修图"
- ResShift 是"带时间步的多步残差修图"

### 四种超分方法对比总表

| | SRResNet | SR3 | ResShift | DDPM（参考） |
|---|----------|-----|----------|-------------|
| 任务 | 超分 | 超分 | 超分 | 无条件生成 |
| 步数 | 1 | 50-100 | ~15 | 500-1000 |
| 预测目标 | 残差 | 噪声 | 残差 | 噪声 |
| 条件输入 | 低清图 | 低清图 | 低清图 | 无 |
| 采样起点 | - | 纯噪声 | 低清图附近 | 纯噪声 |
| 训练 loss | L1 | MSE（噪声） | MSE（残差） | MSE（噪声） |
| 评估指标 | PSNR | PSNR | PSNR | Noise Loss |
| 代码文件 | `trainer/sr.py` | `trainer/sr3.py` | `trainer/resshift.py` | `trainer/diffusion.py` |

---

## 这版实现和官方版的差异

差异需要先讲清楚，不然很容易把目标和官方实现混在一起。

### 1. 数据集差异

| | 官方实现 | 本仓库 |
|---|---------|--------|
| 数据集 | 高分辨率自然图像（DIV2K 等） | MNIST / FashionMNIST / CIFAR10 |
| 图像尺寸 | 通常 256x256 或更大 | 28x28 或 32x32 |
| 通道数 | RGB 3 通道 | 灰度 1 通道 |

### 2. 退化模型差异

| | 官方实现 | 本仓库 |
|---|---------|--------|
| 退化流程 | blur + resize + noise + JPEG 压缩等复杂退化 | 仅 bicubic 下采样 + 可选少量噪声 |
| 目的 | 追求真实恢复效果 | 先讲清楚训练接口和条件恢复逻辑 |

### 3. 网络规模差异

| | 官方实现 | 本仓库 |
|---|---------|--------|
| 网络 | 更大更深的 UNet | 小型 UNet |
| 适用场景 | GPU 训练，追求效果 | CPU / 单卡演示，快速理解代码路径 |

### 4. 目标差异

| | 官方实现 | 本仓库 |
|---|---------|--------|
| 追求 | 真实恢复效果和速度表现 | 是否能看清残差迁移、读懂代码配合、继续扩展 |
| 定位 | 研究级实现 | 教学版骨架 |

---

## 推荐理解顺序

推荐按下面顺序：

1. **先讲 SRResNet**：输入 bicubic 图，直接学残差 → 理解"一步回归"
2. **再讲 SR3**：把 DDPM 变成条件扩散超分 → 理解"多步条件去噪"
3. **最后讲 ResShift**：不是一次性回归，也不完全照搬标准条件扩散，而是少步数逐步恢复残差 → 理解"少步残差恢复"

也可以把三者浓缩成下面三句话：

- **SRResNet**：一步残差修复
- **SR3**：标准条件扩散超分
- **ResShift**：多步围绕条件图恢复残差

最重要的一句可以直接记住：

> SRResNet 是"一步修图"，ResShift 是"带时间步的多步残差修图"。

---

## 运行命令与预期输出

```bash
# 训练 ResShift
python main.py --config configs/sr/resshift.yaml

# 如果没有 GPU
python main.py --config configs/sr/resshift.yaml --device cpu
```

预期输出：

```
Loading config from: configs/sr/resshift.yaml
Using seed: 42
Using device: cuda
Initializing trainer: resshift (ResShiftTrainer)
Epoch [1] Train Residual MSE: 0.123456
Epoch [1] Val PSNR: 19.5678 dB
Saved super-resolution samples to checkpoints/resshift_mnist_toy/sr_epoch_1.png
...
```

> **提示**：ResShift 训练收敛较快，10 个 epoch 通常就能看到明显效果。推理只需 15 步，速度远快于 SR3 和 DDPM。

---

## 常见问题

**Q: ResShift 的 noise_level 参数怎么理解？**
A: `noise_level` 控制中间态中噪声的最大强度。`noise_scales = (1 - residual_scales) * noise_level`，所以噪声强度随残差减少而增大，但不会超过 `noise_level`。值越大，中间态噪声越多，模型需要更强的去噪能力；值越小，中间态更接近低清图，模型更容易但可能恢复不够精细。默认 0.15 是一个经验值。

**Q: 为什么 ResShift 推理时最后一步不加噪声？**
A: 因为 `step=0` 对应 `residual_scale=1.0`，此时模型应该已经能精确预测完整残差。加噪声反而会破坏最终结果。这和 DDPM 中 `t=0` 时不加噪声的逻辑一致。

**Q: 可以把 ResShift 用在去噪或去模糊任务上吗？**
A: 原则上可以。官方论文就把它用在了多种恢复任务上。关键变化是条件输入从"低清图"变成"带噪图"或"模糊图"，残差定义也相应改变。本仓库目前只实现了超分，但框架设计上已经预留了扩展空间。

---

## 下一步扩展

如果后面想把它进一步补完整，可以继续加这几项：

1. **增加真实 SR 数据支持**：`DIV2K / Set5 / Set14` 数据集
2. **扩展退化流程**：从单纯 bicubic 扩展到 blur + resize + noise + jpeg
3. **增加更多指标**：LPIPS、SSIM 等
4. **增加单独的超分推理脚本**：类似 `inference_vae.py` 的 `inference_sr.py`
5. **把简化版 ResShift 再往官方实现靠近**：增加更复杂的调度策略和网络设计
6. **尝试不同的 schedule**：对比 linear 和 cosine 的效果差异
7. **调整 timesteps 和 noise_level**：找到效率与质量的最佳平衡点
