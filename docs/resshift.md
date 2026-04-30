# ResShift 学习说明

## 为什么学 ResShift

ResShift 适合放在学习路径里，不是因为它更复杂，而是因为它把扩散式图像恢复里一个关键的工程问题说得很直接：

传统扩散恢复模型效果强，但推理步数多。对超分、去噪、去 JPEG 这类恢复任务来说，输入已经带了大量结构信息，没必要从纯噪声慢慢采样。

ResShift 的核心思路就是围绕"退化图像到目标图像之间的残差"，设计一个更短、更贴近恢复任务本质的迁移过程。

---

## 官方方法的核心

参考论文和官方仓库，ResShift 强调几件事：

- 把图像恢复看成 residual shifting 过程，不是从纯噪声生成
- 条件输入是退化图像，模型关注的是逐步恢复目标残差
- 比传统扩散恢复方法用更少的采样步数（通常 15 步）
- 官方用于 super-resolution、deblurring、de-noising、face restoration 等多类恢复任务

论文：<https://arxiv.org/abs/2307.12348>
官方仓库：<https://github.com/zsyOAOA/ResShift>

---

## SR3 之后为什么还要学 ResShift

SR3 证明了扩散模型可以带条件做超分。但 ResShift 追问的是另一件事：

对恢复任务来说，真的有必要像无条件生成那样从纯噪声慢慢采样很多步吗？

低质量输入本身已经包含大量结构信息。ResShift 的想法是不要再把问题表述成"从纯噪声生成整张图"，而是"围绕条件图逐步恢复目标残差"。

SR3 让你理解条件扩散怎么成立，ResShift 让你理解条件扩散怎么进一步贴近恢复任务。

---

## 本仓库的实现

这里是简化版 ResShift，目标不是复现论文全部细节，而是把下面这条链路讲清楚：

1. 构造低清图 `LR_up`
2. 计算残差 `R = HR - LR_up`
3. 在不同时间步对残差施加不同强度的保留和噪声扰动
4. 让网络学：给定当前状态 `x_t` 和条件图 `LR_up`，预测残差 `R`
5. 推理时用少量 step 逐步恢复残差，重建高分结果

| 文件 | 职责 |
|------|------|
| `models/resshift.py` | `ResShiftUNet`（网络）+ `ResidualShiftScheduler`（调度器） |
| `trainer/resshift.py` | 训练逻辑：构造中间态 → 预测残差 → 算 loss |
| `configs/sr/resshift.yaml` | 实验配置：timesteps、noise_level、schedule |

如果还没掌握 VAE、DDPM 和 SR3 的基本概念，建议先看 `docs/generative_basics.md`。

---

## 算法详解

### 残差迁移

核心公式：

```
shifted = condition + residual_scale[t] * residual + noise_scale[t] * noise
```

- `condition`：低清条件图 `LR_up`（固定）
- `residual`：`HR - LR_up`（目标残差）
- `residual_scale[t]`：残差保留比例，从 1.0 降到 0.0
- `noise_scale[t]`：噪声强度，随残差减少而增大

直觉：
- `t=0` 时：`shifted ≈ LR_up + 1.0 * residual = HR`，接近目标图
- `t=T` 时：`shifted ≈ LR_up + noise`，接近低清图加噪声

ResShift 的"扩散"不是从纯噪声到图像，而是从"低清图加噪声"到"高清图"。

### 调度器构造

`ResidualShiftScheduler` 初始化时预计算两个数组：

```python
# models/resshift.py
if schedule == "linear":
    residual_scales = torch.linspace(1.0, 0.0, timesteps)  # 线性衰减
elif schedule == "cosine":
    residual_scales = torch.cos(torch.linspace(0.0, pi/2, timesteps))  # 余弦衰减

noise_scales = (1.0 - residual_scales) * noise_level
```

`linear` 匀速衰减，`cosine` 前期慢后期快。`linear` 像匀速刹车，`cosine` 像渐进刹车——开始轻踩，后面越来越重。

### 训练过程

`trainer/resshift.py` 的 `train_one_epoch()`：

```python
def train_one_epoch(self, epoch):
    self.model.train()
    for lr, hr in self.train_loader:
        lr, hr = lr.to(self.device), hr.to(self.device)

        # 随机采样时间步
        batch_size = lr.size(0)
        t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device)

        # 构造中间态和目标残差
        shifted, residual = self.scheduler.q_sample(hr, lr, t)

        # 模型预测残差
        predicted_residual = self.model(torch.cat([shifted, lr], dim=1), t)

        # MSE loss
        loss = self.criterion(predicted_residual, residual)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

几个关键点：
- 模型输入是中间态和条件图的通道拼接
- 训练目标是残差 `HR - LR_up`，不是噪声
- 每个样本的时间步随机采样，模型要学在所有时间步上预测残差

### 推理过程

`models/resshift.py` 的 `ResidualShiftScheduler.sample()`：

```python
@torch.no_grad()
def sample(self, model, condition, clamp_range=None):
    batch_size = condition.shape[0]
    device = condition.device

    # 从低清图附近开始（不是纯噪声）
    noise_scale = self.noise_scales[-1].to(device)
    current = condition + noise_scale * torch.randn_like(condition)

    # 逐步恢复
    for step in reversed(range(self.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        predicted_residual = model(torch.cat([current, condition], dim=1), t)

        # 最后一步直接用预测残差重建
        if step == 0:
            current = condition + predicted_residual
            continue

        # 中间步骤：缩放残差 + 加噪声
        prev_t = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
        residual_scale = self._extract(self.residual_scales, prev_t, current.shape)
        noise_scale = self._extract(self.noise_scales, prev_t, current.shape)
        current = condition + residual_scale * predicted_residual
        current = current + noise_scale * torch.randn_like(current)

    if clamp_range is not None:
        current = current.clamp(*clamp_range)
    return current
```

- 采样起点是 `condition + noise`，不是纯随机噪声
- 每步都把 condition（低清图）当锚点，预测的残差叠加在低清图上
- 最后一步不加噪声，直接用预测残差重建
- 整个过程只需 15 步

---

## 和其他模型的区别

### ResShift vs VAE

VAE 走潜空间：编码 → 采样 → 解码。ResShift 在像素空间里围绕条件图做带时间步的恢复。不走潜空间这条路。

### ResShift vs DDPM

| | DDPM | ResShift |
|---|------|----------|
| 建模对象 | 噪声 ε | 残差 R = HR - LR |
| 采样起点 | 纯随机噪声 | 低清图 + 少量噪声 |
| 采样步数 | 500-1000 | ~15 |
| 条件信息 | 无 | 低清图 |
| 训练目标 | 预测噪声 | 预测残差 |

DDPM 从纯噪声"构建"图像，ResShift 从低清图"修补"图像。

### ResShift vs SR3

| | SR3 | ResShift |
|---|-----|----------|
| 训练目标 | 预测噪声 | 预测残差 |
| 采样起点 | 纯随机噪声 | 低清图附近 |
| 采样步数 | 50-100 | ~15 |
| 中间态构造 | 标准加噪 | 残差缩放 |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |

SR3 解决"扩散怎么做超分"，ResShift 解决"扩散超分怎么更贴近恢复任务、更高效"。

### ResShift vs SRResNet

SRResNet 一步回归：输入低清图，一次前向传播直接输出超分结果。快但表达能力受限于单次前向。

ResShift 多步恢复：根据时间步构造中间态，让模型预测残差，逐步往高分图方向拉。慢一点但更精细。

### 四种超分方法总表

| | SRResNet | SR3 | ResShift | DDPM（参考） |
|---|----------|-----|----------|-------------|
| 步数 | 1 | 50-100 | ~15 | 500-1000 |
| 预测目标 | 残差 | 噪声 | 残差 | 噪声 |
| 条件输入 | 低清图 | 低清图 | 低清图 | 无 |
| 采样起点 | - | 纯噪声 | 低清图附近 | 纯噪声 |
| 训练 loss | L1 | MSE（噪声） | MSE（残差） | MSE（噪声） |
| 评估指标 | PSNR | PSNR | PSNR | Noise Loss |

---

## 从 SR3 改造成 ResShift

### 第一步：改训练目标

把 SR3 的"预测噪声"改成"预测残差"：

1. 保留网络输入形式 `[x_t, LR_up]`
2. 新增 `residual = HR - LR_up`
3. 暂时不改网络骨架
4. loss 改成 `MSE(predicted_residual, residual)`

关键是区分"监督目标"和"模型结构"是两层不同问题。

### 第二步：改前向过程

不再沿用 SR3 的 `q_sample(HR, t, noise)`，改成围绕条件图的 shifted 过程：

`x_t = LR_up + s_t * (HR - LR_up) + sigma_t * z`

然后想清楚两个问题：
- `s_t = 1`、`sigma_t = 0` 时，`x_t` 是什么？
- `s_t = 0`、噪声较大时，`x_t` 更接近什么？

### 第三步：改推理循环

不再通过标准 DDPM 公式从 `predicted_noise` 还原 `x_{t-1}`，改成通过 `predicted_residual` 逐步更新：

`x_{t-1} = LR_up + s_{t-1} * predicted_residual + sigma_{t-1} * z`

### 第四步：理解"为什么能少步"

不少人一开始会问：既然 ResShift 更快，是不是只要把 step 改少就行？

不是。正确顺序是：先改问题定义 → 再改训练目标 → 再改 forward/reverse → 最后讨论"少步采样为什么合理"。否则会把 ResShift 误解成"只是缩短版 SR3"。

---

## 和官方版的差异

| | 官方实现 | 本仓库 |
|---|---------|--------|
| 数据集 | DIV2K 等高分辨率自然图像 | MNIST / FashionMNIST / CIFAR10 |
| 图像尺寸 | 256x256 或更大 | 28x28 或 32x32 |
| 退化流程 | blur + resize + noise + JPEG | 仅 bicubic 下采样 + 可选少量噪声 |
| 网络规模 | 更大更深的 UNet | 小型 UNet |
| 目标 | 真实恢复效果 | 看清残差迁移、读懂代码、继续扩展 |

---

## 运行命令

```bash
python main.py --config configs/sr/resshift.yaml
# 没有 GPU：
python main.py --config configs/sr/resshift.yaml --device cpu
```

ResShift 训练收敛较快，10 个 epoch 通常就能看到明显效果。

---

## 常见问题

**`noise_level` 参数怎么理解？** 它控制中间态噪声的最大强度。`noise_scales = (1 - residual_scales) * noise_level`，噪声随残差减少而增大，但不超过 `noise_level`。值越大模型需要更强的去噪能力，值越小中间态更接近低清图。默认 0.15 是经验值。

**为什么最后一步不加噪声？** `step=0` 对应 `residual_scale=1.0`，此时模型应该能精确预测完整残差。加噪声反而破坏结果。和 DDPM 中 `t=0` 不加噪声的逻辑一致。

**能用在去噪或去模糊上吗？** 原则上可以。官方论文就用在了多种恢复任务上。关键变化是条件输入从"低清图"变成"带噪图"或"模糊图"，残差定义也相应改变。本仓库目前只实现了超分。

---

## 练习

### 题 1：概念判断

用自己的话回答：
1. ResShift 不是"把 SR3 的 step 从 1000 改成 15"这么简单，为什么？
2. 超分任务里预测残差通常比从纯噪声生整图更贴近任务，为什么？
3. SR3 和 ResShift 的共同点是什么，根本差异是什么？

### 题 2：输入输出对比

写出 SR3 和 ResShift 的训练时输入输出，至少带一版符号表达式。

### 题 3：从 SR3 改造 trainer

以 `trainer/sr3.py` 为起点，列出改造成 `trainer/resshift.py` 的最小步骤。定位：哪段代码负责采样时间步、构造 `x_t`、定义监督目标、推理循环。

### 题 4：手推 shifted 公式

给定 `HR = 0.9`、`LR_up = 0.5`、`s_t = 0.25`、`sigma_t = 0.1`、`z = -0.4`，手算残差 `R` 和当前状态 `x_t`。然后回答：`s_t` 从 0.25 增大到 0.75 时，`x_t` 往哪个方向变化？

### 题 5：读代码定位

在仓库里定位：ResShift 的残差比例 `s_t`、噪声比例 `sigma_t`、训练时的 `predicted_residual`、推理时的最终重建分别在哪个文件、哪个函数。

### 题 6：对比实验

把 `timesteps` 改成 5 / 15 / 30 三组，对比推理速度和可视化结果，解释为什么 ResShift 能在较少 step 下保留恢复能力。

---

## 自查清单

学完后至少能独立说清：

- SR3 输出噪声预测，ResShift 输出残差预测
- SR3 更像标准条件扩散，ResShift 更像面向恢复任务的残差迁移
- 从工程实现看，ResShift 主要改 trainer 和 scheduler，不一定先改 UNet
- ResShift 的少步采样来自任务表述变化，不是"粗暴减少 step 数"

---

## 下一步扩展

1. 增加真实 SR 数据支持（DIV2K / Set5 / Set14）
2. 扩展退化流程（blur + resize + noise + jpeg）
3. 增加更多指标（LPIPS、SSIM）
4. 增加单独的超分推理脚本 `inference_sr.py`
5. 往官方实现靠近：更复杂的调度策略和网络设计
6. 对比 linear 和 cosine schedule 的效果差异
7. 调整 timesteps 和 noise_level，找效率与质量的平衡点
