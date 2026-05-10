# ResShift 学习说明

## 为什么学 ResShift

ResShift 适合放在学习路径里，不是因为它更复杂，而是因为它把扩散式图像恢复里一个关键的工程问题说得很直接：

传统扩散恢复模型效果强，但推理步数多。对超分、去噪、去 JPEG 这类恢复任务来说，输入已经带了大量结构信息，没必要从纯噪声慢慢采样。

ResShift 的核心思路就是围绕退化图像和目标图像之间的 shifting 过程，设计一个更短、更贴近恢复任务本质的迁移链。

---

## 官方方法的核心

参考论文和官方仓库，ResShift 强调几件事：

- 把图像恢复看成 residual shifting 过程，不是从纯噪声生成
- 条件输入是退化图像，模型关注的是逐步恢复目标图像
- 比传统扩散恢复方法用更少的采样步数（通常 15 步）
- 官方用于 super-resolution、deblurring、de-noising、face restoration 等多类恢复任务

论文：<https://arxiv.org/abs/2307.12348>
官方仓库：<https://github.com/zsyOAOA/ResShift>

---

## SR3 之后为什么还要学 ResShift

SR3 证明了扩散模型可以带条件做超分。但 ResShift 追问的是另一件事：

对恢复任务来说，真的有必要像无条件生成那样从纯噪声慢慢采样很多步吗？

低质量输入本身已经包含大量结构信息。ResShift 的想法是不要再把问题表述成"从纯噪声生成整张图"，而是"围绕条件图逐步恢复目标图像"。

SR3 让你理解条件扩散怎么成立，ResShift 让你理解条件扩散怎么进一步贴近恢复任务。

> SR3 和 ResShift 的知识点对比也收录在 [生成模型知识补充](generative_basics.md#resshift)。

---

## 本仓库的实现

这里是简化版 ResShift，目标不是复现论文全部细节，而是把下面这条链路讲清楚：

1. 构造低清图 `LR_up`
2. 构造论文里的 forward state `x_t`
3. 让网络学：给定当前状态 `x_t`、条件图 `LR_up` 和时间步 `t`，预测 `x_0 = HR`
4. 推理时从 `LR_up + noise` 开始，按 posterior 逐步恢复 `x_0`
5. 用较少 step 完成采样

| 文件 | 职责 |
|------|------|
| [models/resshift.py](../models/resshift.py) | `ResShiftUNet`（网络）+ `ResidualShiftScheduler`（调度器） |
| [trainer/resshift.py](../trainer/resshift.py) | 训练逻辑：构造中间态 → 预测 `x_0` → 算 loss |
| [configs/sr/resshift.yaml](../configs/sr/resshift.yaml) | 实验配置：timesteps、noise_level、schedule |

如果还没掌握 VAE、DDPM 和 SR3 的基本概念，建议先看 [生成模型知识补充](generative_basics.md)。

---

## 算法详解

### Residual Shifting

核心公式：

```
x_t = x_0 + eta_t (y_0 - x_0) + kappa * sqrt(eta_t) * noise
```

- `x_0`：目标 HR 图
- `y_0`：上采样到 HR 尺寸后的 LR 条件图
- `eta_t`：shifting 序列，随时间步增大
- `kappa * sqrt(eta_t)`：论文里的噪声强度

直觉：
- `t` 很小时：`x_t ≈ x_0`，接近目标 HR
- `t` 很大时：`x_t ≈ y_0 + noise`，接近低清图附近的先验状态

ResShift 的关键不是“预测残差”，而是 forward / reverse Markov chain 围绕 `(x_0, y_0)` 之间的 residual shifting 来设计。

### 调度器构造

`ResidualShiftScheduler` 初始化时预计算 `eta_t` 和 reverse posterior variance：

```python
eta_t = ...
alpha_t = eta_t - eta_{t-1}
posterior_variance_t = kappa^2 * (eta_{t-1} / eta_t) * alpha_t
```

当前默认使用论文风格的 `geometric` 调度，并暴露 `shift_power` 控制增长速度。

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

        # 构造中间态和目标 x0
        shifted, target_x0 = self.scheduler.q_sample(hr, lr, t)

        # 模型预测 x0
        predicted_x0 = self.model(torch.cat([shifted, lr], dim=1), t)

        # MSE loss
        loss = self.criterion(predicted_x0, target_x0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

几个关键点：
- 模型输入是中间态和条件图的通道拼接
- 训练目标是 `x_0 = HR`，不是噪声，也不是 residual
- 每个样本的时间步随机采样，模型要学在所有时间步上恢复 `x_0`

### 推理过程

`models/resshift.py` 的 `ResidualShiftScheduler.sample()`：

```python
@torch.no_grad()
def sample(self, model, condition, clamp_range=None):
    batch_size = condition.shape[0]
    device = condition.device

    # 从 y0 + kappa * noise 开始
    current = condition + self.kappa * torch.randn_like(condition)

    # 逐步恢复
    for step in reversed(range(self.timesteps)):
        t = torch.full((batch_size,), step, device=device, dtype=torch.long)
        predicted_x0 = model(torch.cat([current, condition], dim=1), t)

        if step == 0:
            current = predicted_x0
        else:
            mean, variance = self.p_mean_variance(current, predicted_x0, t)
            current = mean + torch.sqrt(variance) * torch.randn_like(current)

    if clamp_range is not None:
        current = current.clamp(*clamp_range)
    return current
```

- 采样起点是 `y_0 + noise`，不是纯随机噪声
- 每步都输入 `(x_t, y_0, t)`，模型输出 `x_0`
- reverse 更新使用论文里的 posterior mean / variance
- 整个过程只需 15 步

---

## 和其他模型的区别

### ResShift vs VAE

VAE 走潜空间：编码 → 采样 → 解码。ResShift 在像素空间里围绕条件图做带时间步的恢复。不走潜空间这条路。

### ResShift vs DDPM

| | DDPM | ResShift |
|---|------|----------|
| 建模对象 | 噪声 ε | residual shifting 过程 |
| 采样起点 | 纯随机噪声 | 低清图 + 少量噪声 |
| 采样步数 | 500-1000 | ~15 |
| 条件信息 | 无 | 低清图 |
| 训练目标 | 预测噪声 | 预测 `x_0` |

DDPM 从纯噪声"构建"图像，ResShift 从低清图"修补"图像。

### ResShift vs SR3

| | SR3 | ResShift |
|---|-----|----------|
| 训练目标 | 预测噪声 | 预测 `x_0` |
| 采样起点 | 纯随机噪声 | 低清图附近 |
| 采样步数 | 50-100 | ~15 |
| 中间态构造 | 标准加噪 | `x_0 -> y_0` shifting |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |

SR3 解决"扩散怎么做超分"，ResShift 解决"扩散超分怎么更贴近恢复任务、更高效"。

### ResShift vs SRResNet

SRResNet 一步回归：输入低清图，一次前向传播直接输出超分结果。快但表达能力受限于单次前向。

ResShift 多步恢复：根据时间步构造中间态，让模型预测 `x_0`，逐步往高分图方向拉。慢一点但更精细。

### 四种超分方法总表

| | SRResNet | SR3 | ResShift | DDPM（参考） |
|---|----------|-----|----------|-------------|
| 步数 | 1 | 50-100 | ~15 | 500-1000 |
| 预测目标 | 残差 | 噪声 | `x_0` | 噪声 |
| 条件输入 | 低清图 | 低清图 | 低清图 | 无 |
| 采样起点 | - | 纯噪声 | 低清图附近 | 纯噪声 |
| 训练 loss | L1 | MSE（噪声） | MSE（`x_0`） | MSE（噪声） |
| 评估指标 | PSNR | PSNR | PSNR | Noise Loss |

---

## 从 SR3 改造成 ResShift

### 第一步：改训练目标

把 SR3 的"预测噪声"改成"预测 `x_0`"：

1. 保留网络输入形式 `[x_t, LR_up]`
2. 监督目标直接改成 `HR`
3. 暂时不改网络骨架
4. loss 改成 `MSE(predicted_x0, hr)`

关键是区分"监督目标"和"模型结构"是两层不同问题。

### 第二步：改前向过程

不再沿用 SR3 的 `q_sample(HR, t, noise)`，改成论文里的 shifting 过程：

`x_t = x_0 + eta_t (y_0 - x_0) + kappa * sqrt(eta_t) * z`

然后想清楚两个问题：
- `eta_t` 很小时，`x_t` 更接近什么？
- `eta_t` 接近 1 时，`x_t` 更接近什么？

### 第三步：改推理循环

不再通过标准 DDPM 公式从 `predicted_noise` 还原 `x_{t-1}`，改成通过 `predicted_x0` 和论文 posterior 逐步更新：

`x_{t-1} ~ N((eta_{t-1}/eta_t) * x_t + (alpha_t/eta_t) * predicted_x0, var_t I)`

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
| 目标 | 真实恢复效果 | 看清 shifting 过程、读懂代码、继续扩展 |

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

**`noise_level` 参数怎么理解？** 它对应论文里的 `kappa`，控制 forward/reverse 过程的总体噪声强度。值越大模型需要更强的去噪能力，值越小中间态更接近低清图。默认 0.15 是教学配置里的经验值。

**为什么最后一步不加噪声？** `step=0` 时直接输出 `predicted_x0`。这对应 reverse 过程的终点，不再采样额外噪声。

**能用在去噪或去模糊上吗？** 原则上可以。官方论文就用在了多种恢复任务上。关键变化是条件输入从"低清图"变成"带噪图"或"模糊图"，而 `y_0` 的定义随任务改变。本仓库目前只实现了超分。

---

## 练习

### 题 1：概念判断

用自己的话回答：
1. ResShift 不是"把 SR3 的 step 从 1000 改成 15"这么简单，为什么？
2. 为什么 ResShift 的 prior 设在低清图附近，会比从纯噪声开始更贴近超分任务？
3. SR3 和 ResShift 的共同点是什么，根本差异是什么？

### 题 2：输入输出对比

写出 SR3 和 ResShift 的训练时输入输出，至少带一版符号表达式。

### 题 3：从 SR3 改造 trainer

以 `trainer/sr3.py` 为起点，列出改造成 `trainer/resshift.py` 的最小步骤。定位：哪段代码负责采样时间步、构造 `x_t`、定义监督目标、推理循环。

### 题 4：手推 shifted 公式

给定 `x_0 = 0.9`、`y_0 = 0.5`、`eta_t = 0.25`、`kappa = 0.2`、`z = -0.4`，手算当前状态 `x_t`。然后回答：`eta_t` 从 0.25 增大到 0.75 时，`x_t` 往哪个方向变化？

### 题 5：读代码定位

在仓库里定位：ResShift 的 `eta_t`、posterior variance、训练时的 `predicted_x0`、推理时的最终重建分别在哪个文件、哪个函数。

### 题 6：对比实验

把 `timesteps` 改成 5 / 15 / 30 三组，对比推理速度和可视化结果，解释为什么 ResShift 能在较少 step 下保留恢复能力。

---

## 自查清单

学完后至少能独立说清：

- SR3 输出噪声预测，ResShift 输出 `x_0` 预测
- SR3 更像标准条件扩散，ResShift 更像面向恢复任务的 residual shifting
- 从工程实现看，ResShift 主要改 trainer 和 scheduler，不一定先改 UNet
- ResShift 的少步采样来自任务表述变化，不是"粗暴减少 step 数"

> 更多扩展练习见 [学习路径 > 工程实践与扩展](learning_path.md#第七阶段工程实践与扩展)。

---

## 下一步扩展

1. 增加真实 SR 数据支持（DIV2K / Set5 / Set14）
2. 扩展退化流程（blur + resize + noise + jpeg）
3. 增加更多指标（LPIPS、SSIM）
4. 增加单独的超分推理脚本 `inference_sr.py`
5. 往官方实现靠近：更复杂的调度策略和网络设计
6. 对比 linear 和 cosine schedule 的效果差异
7. 调整 timesteps 和 noise_level，找效率与质量的平衡点
