# 生成模型知识补充

它不追求把公式讲得很满，而是重点回答三个问题：

1. VAE、DDPM、SR3、ResShift 分别在解决什么问题
2. 它们的训练目标为什么不一样
3. 它们在本项目里分别对应哪些代码

---

## 先区分两类任务

### 1. 判别任务

分类任务属于判别任务。

它关心的是：

- 输入一张图
- 判断它属于哪一类

例如：

- 输入手写数字图像
- 输出 0 到 9 的类别

这类任务的目标是"做判断"。

**代码体现**：`trainer/base.py` 中 `train_one_epoch()` 的核心逻辑：

```python
logits = self.model(x)          # 前向：图像 → 类别概率
loss = self.criterion(logits, y) # 计算：预测和标签的差距
loss.backward()                  # 反向传播
self.optimizer.step()            # 更新参数
```

### 2. 生成任务

VAE、DDPM、SR3、ResShift 都属于生成建模或恢复建模。

它们关心的是：

- 如何生成一张合理的图像
- 或者如何从一个退化状态恢复出更好的图像

这类任务的目标不是分类，而是"建模数据分布"或者"重建目标图像"。

---

## 为什么生成模型比分类难

分类通常只需要输出一个类别编号。

生成模型需要回答的问题更难：

- 什么样的像素组合才像真实图像
- 图像的整体结构和局部细节如何同时合理
- 如果输入里有噪声或退化，应该恢复到什么程度

所以在工程上，生成模型通常会比分类任务多出下面这些概念：

| 概念 | 出现在哪个模型 | 直觉理解 |
|------|---------------|----------|
| 潜变量 `z` | VAE | 图像在低维空间中的"压缩表示" |
| 重参数化 | VAE | 让随机采样变得可导的技巧 |
| 时间步 `t` | DDPM / SR3 / ResShift | 表示当前处于加噪/恢复过程的哪个阶段 |
| 噪声调度 | DDPM / SR3 | 控制每一步加噪的强度 |
| 条件输入 | SR3 / ResShift | 给模型提供额外信息（如低清图） |
| 采样过程 | DDPM / SR3 / ResShift | 推理时从噪声/中间态逐步生成/恢复图像 |

---

## VAE 是什么

VAE 的全称是 Variational AutoEncoder，中文通常叫"变分自编码器"。

它最直观的理解方式是：

- 编码器把图像压缩到一个低维潜空间
- 解码器再从潜空间把图像重建出来

和普通 AutoEncoder 不同的是，VAE 不直接输出一个确定的隐藏向量，而是输出一个分布。

也就是说，编码器不是直接说"这张图对应向量 `z`"，而是说：

- 这张图的潜变量均值是 `mu`
- 方差相关参数是 `logvar`

然后再从这个分布里采样出 `z`，交给解码器生成图像。

### VAE 为什么要学分布

因为生成模型最终希望做到：

- 不只会重建训练集里的样本
- 还能够从潜空间中随机采样，生成新样本

如果潜空间没有被约束成比较平滑、连续的分布，那么随机采样出来的 `z` 往往没有意义。

**打个比方**：想象潜空间是一个地图。如果地图上只有几个孤立的"城市"（训练样本的位置），那随机扔一个飞镖大概率落在荒野。KL 散度的作用就是把城市"铺开"，让整个地图都有意义。

### VAE 的 loss 在做什么

VAE 常见 loss 可以拆成两部分：

- **重构损失**：希望解码后的图像接近原图
- **KL 损失**：希望潜变量分布接近标准正态分布

你可以把它理解成：

- 重构损失负责"把图还原对"
- KL 损失负责"把潜空间整理好"

如果只有重构损失，模型容易变成普通自编码器——能重建但不能生成。

如果 KL 太强，模型又可能重构得太差——潜空间很整齐但图像模糊。

所以 VAE 的训练本质上是在这两者之间做平衡。

**代码对应**：`trainer/vae.py` 中的 `_build_criterion()`：

```python
def vae_loss(x_recon, x, mu, logvar):
    # 重构损失：解码图和原图的差异
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    # KL 损失：潜变量分布离标准正态的距离
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
```

### 重参数化为什么重要

采样操作本身不可直接反向传播。

VAE 通过重参数化技巧，把

- `z ~ N(mu, sigma^2)`

改写成

- `z = mu + eps * sigma`，其中 `eps ~ N(0, 1)`

这样梯度就可以通过 `mu` 和 `sigma` 传播回编码器。

**打个比方**：想象你要从一个"会动的靶子"上采样。直接采样的话，你不知道靶子是怎么动的，梯度传不回去。重参数化相当于说"靶子的中心是 mu，晃动幅度是 sigma，随机风是 eps"，这样你就知道该怎么调整中心和幅度了。

**代码对应**：`models/vae.py` 中的 `reparameterize()`：

```python
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)   # sigma = exp(0.5 * logvar)
    eps = torch.randn_like(std)      # eps ~ N(0, 1)，随机噪声
    return mu + eps * std            # z = mu + eps * sigma，可导！
```

### 本项目里 VAE 对应哪里

| 文件 | 看什么 |
|------|--------|
| `models/vae.py` | 编码器输出 `mu` 和 `logvar`；`reparameterize()` 把采样写成可训练形式 |
| `trainer/vae.py` | loss 为什么返回三项（总 loss / 重构 / KL）；`_monitor_mode()` 返回 `"min"` |
| `configs/generate/vae.yaml` | `latent_dim: 20` 控制潜空间维度 |
| `inference_vae.py` | 如何从潜空间随机采样并生成图像 |

### 容易混淆的点

- VAE 不是直接记忆训练图像，而是在学习一个潜空间分布
- VAE 的生成不是从纯像素空间硬生成，而是先采样潜变量再解码
- VAE 的验证指标常常是 loss，而不是分类准确率
- VAE 生成的图像通常比扩散模型模糊，因为 VAE 优化的是"平均"输出

---

## DDPM 是什么

DDPM 的全称是 Denoising Diffusion Probabilistic Model。

直观理解可以分成两半：

### 1. 正向扩散

把一张真实图像逐步加噪。

一步一步之后，图像会越来越像高斯噪声。

```
原图 x_0 → 加噪 x_1 → 加噪 x_2 → ... → 纯噪声 x_T
```

**代码对应**：`models/ddpm/diffusion.py` 中的 `q_sample()`：

```python
def q_sample(self, x_start, t, noise=None):
    # x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
    if noise is None: noise = torch.randn_like(x_start)
    return (self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
```

### 2. 反向去噪

训练一个网络，让它学会在每个时间步把噪声去掉一点。

如果这个过程学好了，那么从纯噪声开始反复去噪，就能得到一张新图像。

```
纯噪声 x_T → 去噪 x_{T-1} → 去噪 x_{T-2} → ... → 生成图 x_0
```

**代码对应**：`models/ddpm/diffusion.py` 中的 `p_sample()`：

```python
def p_sample(self, model, x_t, t, clip_denoised_range=(-1.0, 1.0)):
    pred_noise = model(x_t, t)           # 模型预测噪声
    return self.p_sample_from_pred_noise( # 从预测噪声推算 x_{t-1}
        x_t, t, pred_noise, clip_denoised_range=clip_denoised_range)
```

### 为什么 DDPM 要引入时间步 `t`

因为不同时间步的图像噪声程度不同。

模型必须知道当前面对的是：

- 轻微加噪的图（`t` 很小，图像还比较清晰）
- 还是已经几乎看不出原图的图（`t` 很大，几乎全是噪声）

所以模型输入通常是：

- 当前噪声图 `x_t`
- 当前时间步 `t`

这也是为什么扩散模型的网络常常需要时间步嵌入。

**代码对应**：`models/ddpm/unet.py` 中的 `timestep_embedding()`：

```python
def timestep_embedding(timesteps, dim, max_period=10000):
    # 把整数 t 编码成 dim 维向量，类似 Transformer 的位置编码
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
```

### DDPM 训练时到底在预测什么

在这个实现里，最常见的做法是预测噪声。

也就是：

1. 先对原图加噪，得到 `x_t`
2. 同时保留这次加进去的噪声 `noise`
3. 让模型输入 `(x_t, t)`
4. 输出对 `noise` 的预测

如果模型能把噪声预测准，就说明它学会了如何把图像从噪声状态拉回真实数据分布。

**代码对应**：`trainer/diffusion.py` 中的 `_compute_noise_prediction_loss()`：

```python
def _compute_noise_prediction_loss(self, images):
    batch_size = images.shape[0]
    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
    noise = torch.randn_like(images)                                    # 真实噪声
    x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise) # 加噪图
    predicted_noise = self.model(x_noisy, t)                            # 预测噪声
    return self.criterion(predicted_noise, noise)                       # MSE loss
```

### DDPM 为什么采样慢

因为它不是一步生成，而是多步去噪。

推理时要做很多轮：

```
1. 从随机噪声开始
2. 预测当前噪声
3. 反推到更干净一点的状态
4. 再继续下一步
```

步数越多，通常表达能力越强，但推理也越慢。

**打个比方**：DDPM 采样就像雕刻——每次只去掉一小片碎屑，需要很多步才能完成。而 ResShift 更像是修补——已经有了一个大致的形状（低清图），只需要几步精修。

### beta schedule 的直觉

`beta schedule` 控制每一步加噪的强度：

| Schedule | 特点 | 直觉 |
|----------|------|------|
| `linear` | beta 从 0.0001 线性增长到 0.02 | 前期加噪慢，后期加噪快 |
| `cosine` | 用余弦函数控制 | 加噪过程更平滑，不会在后期突然加太多噪声 |

**代码对应**：`models/ddpm/diffusion.py` 中的 `__init__()`：

```python
if beta_schedule == 'linear':
    betas = torch.linspace(0.0001, 0.02, timesteps)
elif beta_schedule == 'cosine':
    betas = cosine_beta_schedule(timesteps)
```

### 本项目里 DDPM 对应哪里

| 文件 | 看什么 |
|------|--------|
| `models/ddpm/unet.py` | `timestep_embedding()` 如何把时间步编码成向量；`UNetModel.forward(x, timesteps)` |
| `models/ddpm/diffusion.py` | `q_sample()` 加噪、`p_sample()` 去噪、`_extract()` 取出对应时间步的系数 |
| `trainer/diffusion.py` | 训练时随机采样时间步 `t`；验证时额外保存采样图片 |
| `configs/generate/ddpm.yaml` | `diffusion.timesteps: 500`、`diffusion.schedule: linear` |

### 容易混淆的点

- DDPM 训练时不是每次都从纯噪声开始训练整个采样链
- 它通常是随机抽一个时间步，训练模型在这个时间步上会"去噪"
- 训练目标往往是噪声预测，不是直接预测类别或直接输出最终图像
- 采样过程只在推理时才做，训练时不需要走完整个链路

---

## SR3 是什么

SR3 可以理解成"把扩散模型引入超分任务"的一个经典桥梁。

这里最值得先理解的不是论文细节，而是下面这件事：

- 扩散模型不只能从纯噪声生成图像
- 也可以在一个条件图像的帮助下，逐步恢复更高质量的结果

### SR3 和 DDPM 的关系

SR3 并不是把 DDPM 推翻重来，而是在 DDPM 的基础上加入条件输入。

所以你可以把两者理解成：

| | DDPM | SR3 |
|---|------|-----|
| 模型输入 | `(x_t, t)` | `([x_t, lr], t)` |
| 训练目标 | 预测噪声 | 预测噪声 |
| 采样起点 | 纯随机噪声 | 纯随机噪声 |
| 条件信息 | 无 | 低清图 `lr` |
| 网络结构 | UNet (in_channels=1) | UNet (in_channels=2) |

也就是说，SR3 的关键变化不是"时间步没了"，而是：

- 模型不再只看当前噪声图
- 还会看低清条件图

**代码对应**：`trainer/sr3.py` 中的 `_predict_noise()`：

```python
def _predict_noise(self, x_noisy, lr, t):
    # 把噪声图和低清条件图在通道维度拼接
    return self.model(torch.cat([x_noisy, lr], dim=1), t)
```

而 `models/sr3.py` 中的 `SR3UNet` 只做了一件事——把 `in_channels` 从 1 改成 2：

```python
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

### SR3 在整条路径里的作用

SR3 是从 DDPM 到 ResShift 之间最重要的桥梁。

如果没有 SR3，很容易觉得：

- 前面学的是无条件生成
- 后面突然跳到了条件恢复

有了 SR3，这条路径就变成：

```
DDPM（无条件扩散）→ SR3（条件扩散）→ ResShift（少步残差恢复）
```

每一步只变一个东西，理解起来更顺畅。

### SR3 推理时的采样过程

SR3 推理时从纯噪声开始，逐步去噪，每一步都参考低清条件图：

```
1. current = 随机噪声
2. for step = T-1, T-2, ..., 0:
3.     predicted_noise = model(cat([current, lr], dim=1), step)
4.     current = diffusion.p_sample_from_pred_noise(current, step, predicted_noise)
5. return current（即超分结果）
```

**代码对应**：`trainer/sr3.py` 中的 `infer()` 方法。

### 本项目里 SR3 对应哪里

| 文件 | 看什么 |
|------|--------|
| `models/sr3.py` | 整个类只有 16 行，核心是 `in_channels=2` |
| `trainer/sr3.py` | `_predict_noise()` 中 `torch.cat([x_noisy, lr], dim=1)` 是条件注入的关键 |
| `models/ddpm/diffusion.py` | SR3 直接复用 DDPM 的 `GaussianDiffusion` |
| `configs/sr/sr3.yaml` | `diffusion.timesteps: 50`（比 DDPM 的 500 少很多） |

### 容易混淆的点

- SR3 不是"一步超分网络"，它仍然是多步采样
- SR3 不是 ResShift，它更接近标准条件扩散超分
- SR3 的关键作用主要在于桥接 DDPM 和 ResShift
- SR3 推理时从纯噪声开始（不是从低清图开始），这是和 ResShift 的重要区别

---

## ResShift 是什么

ResShift 可以理解成一种"更贴近图像恢复任务"的扩散思路。

它和无条件 DDPM 最大的不同是：

- DDPM 常常从纯噪声开始生成
- ResShift 有一个条件图像，比如低清图 `LR_up`

对超分任务来说，输入图像已经包含了大量结构信息。

所以问题不再是"从零生成一张图"，而是：

- 已经有一个低质量版本
- 如何把它逐步修复成高质量版本

### 为什么 ResShift 要关注残差

超分里常见一个思想：

- 低清图里已经有大部分低频结构（整体形状、明暗分布）
- 真正难补的是高频细节（边缘锐度、纹理清晰度）

所以模型不一定非得重新生成整张图，更合理的做法是学习：

- `残差 = 高分图 - 低清图`

这样网络更聚焦于"缺失的信息"。

**打个比方**：低清图就像一幅素描草稿，残差就是上色的部分。你不需要重新画草稿，只需要在草稿上添色。

### ResShift 的训练过程

ResShift 的训练核心是构造"中间态"：

```
1. 计算残差：residual = HR - LR_up
2. 采样时间步 t
3. 构造中间态：shifted = LR_up + residual_scale[t] * residual + noise_scale[t] * noise
4. 让模型预测残差：predicted_residual = model(cat([shifted, LR_up], dim=1), t)
5. 计算 loss：MSE(predicted_residual, residual)
```

其中 `residual_scale` 从 1.0 逐步降到 0.0，`noise_scale` 相应增大。这意味着：

- `t=0` 时：中间态接近目标图（残差保留完整）
- `t=T` 时：中间态接近低清图加噪声（残差几乎消失）

**代码对应**：`models/resshift.py` 中的 `ResidualShiftScheduler.q_sample()`：

```python
def q_sample(self, target, condition, t, noise=None):
    residual = target - condition
    if noise is None: noise = torch.randn_like(target)
    residual_scale = self._extract(self.residual_scales, t, target.shape)
    noise_scale = self._extract(self.noise_scales, t, target.shape)
    shifted = condition + residual_scale * residual + noise_scale * noise
    return shifted, residual
```

### ResShift 的推理过程

推理时从低清图附近开始，逐步恢复残差：

```
1. current = LR_up + noise_scale[-1] * 随机噪声
2. for step = T-1, T-2, ..., 1:
3.     predicted_residual = model(cat([current, LR_up], dim=1), step)
4.     current = LR_up + residual_scale[step-1] * predicted_residual + noise_scale[step-1] * 噪声
5. current = LR_up + predicted_residual  (step=0 时不加噪声)
6. return current
```

**代码对应**：`models/resshift.py` 中的 `ResidualShiftScheduler.sample()`。

### ResShift 和 DDPM 的区别

可以这样粗略理解：

| | DDPM | ResShift |
|---|------|----------|
| 建模对象 | 噪声 | 残差 |
| 采样起点 | 纯随机噪声 | 低清图 + 少量噪声 |
| 采样步数 | 通常 500-1000 | 通常 15 |
| 条件信息 | 无 | 低清图 |
| 训练目标 | 预测噪声 ε | 预测残差 R |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |

### ResShift 和 SR3 的区别

SR3 已经说明了"条件扩散可以做超分"。

ResShift 再往前走一步，强调的是：

- 在恢复任务里，条件图本身已经有大量结构信息
- 所以可以围绕残差设计更高效的少步恢复过程

| | SR3 | ResShift |
|---|-----|----------|
| 训练目标 | 预测噪声 | 预测残差 |
| 采样起点 | 纯随机噪声 | 低清图附近 |
| 采样步数 | 通常 50-100 | 通常 15 |
| 中间态构造 | 标准加噪 `x_t = √ᾱ_t·HR + √(1-ᾱ_t)·ε` | 残差缩放 `shifted = LR + scale·R + noise·ε` |
| 调度器 | `GaussianDiffusion` | `ResidualShiftScheduler` |

### ResShift 和 SRResNet 的区别

SRResNet 是一步回归。

也就是说：

- 输入低清图
- 网络一次前向传播
- 直接输出超分结果

ResShift 则是多步恢复。

它会：

1. 根据时间步构造一个中间状态
2. 让模型预测残差
3. 逐步把结果往高分图方向拉近

所以可以把它理解成：

- SRResNet：一步修图（快但可能不够精细）
- ResShift：多步残差修图（慢一点但更精细）

### 本项目里的 ResShift 在做什么

本项目里的简化版 ResShift 主要保留三件事：

- 条件输入是低清图
- 学的是 `HR - LR_up` 残差
- 推理步数少于传统 DDPM

它没有完整复现论文里的所有细节，而是保留最核心的主线。

### 本项目里 ResShift 对应哪里

| 文件 | 看什么 |
|------|--------|
| `models/resshift.py` | `ResidualShiftScheduler.q_sample()` 如何构造训练中间态；`sample()` 如何做逐步恢复 |
| `trainer/resshift.py` | 训练时 `scheduler.q_sample(hr, lr, t)` 构造中间态；推理时 `scheduler.sample(model, lr)` |
| `configs/sr/resshift.yaml` | `resshift.timesteps: 15`、`noise_level: 0.15`、`schedule: cosine` |

---

## 四者怎么放在一张图里理解

可以用下面这张心智图：

### VAE

- **路线**：图像 → 潜变量 → 图像
- **关键词**：编码、解码、潜空间、KL 散度
- **优点**：结构清晰，采样快，适合讲潜变量建模
- **局限**：生成质量通常不如强扩散模型
- **训练目标**：重构损失 + KL 损失
- **采样方式**：采样 z → 解码，一步完成

### DDPM

- **路线**：图像 ↔ 多步噪声过程
- **关键词**：时间步、噪声调度、去噪采样
- **优点**：生成质量强，扩散思想清楚
- **局限**：推理慢（需要几百步去噪）
- **训练目标**：噪声预测 MSE
- **采样方式**：T 步迭代去噪

### SR3

- **路线**：低清条件图 + 多步扩散恢复 → 高清图
- **关键词**：条件扩散、超分恢复、噪声预测
- **优点**：是 DDPM 到 ResShift 的天然桥梁
- **局限**：推理仍然偏慢
- **训练目标**：噪声预测 MSE（和 DDPM 一样）
- **采样方式**：T 步条件去噪

### ResShift

- **路线**：低清图 + 多步残差恢复 → 高清图
- **关键词**：条件恢复、残差迁移、少步采样
- **优点**：更贴近图像恢复任务，容易连接 SR 和扩散
- **局限**：本仓库是简化版，不等于官方完整实现
- **训练目标**：残差预测 MSE
- **采样方式**：少步残差恢复（约 15 步）

### 四者演进关系

```
VAE（潜空间建模）
 ↓ 引入时间步
DDPM（逐步加噪/去噪）
 ↓ 引入条件输入
SR3（条件扩散超分）
 ↓ 围绕残差简化过程
ResShift（少步残差恢复）
```

每一步只变一个核心概念，这就是这条学习路径的设计意图。

---

## 为什么这个项目要同时讲 VAE 和扩散

因为它们代表了两种非常典型的生成建模思路。

### VAE 教会你什么

- 什么是潜变量
- 什么是概率分布约束
- 什么是重参数化
- 为什么一个 loss 可以由多个部分组成

### 扩散模型教会你什么

- 什么是显式的时间步建模
- 什么是逐步变换过程
- 为什么训练目标可以是噪声而不是图像类别
- 为什么采样过程也属于模型设计的一部分

换句话说：

- VAE 更适合讲"潜空间"
- DDPM / ResShift 更适合讲"过程建模"

---

## 和本项目工程结构怎么对应

最好把"知识点"和"代码分层"一起看。

### 当你学 VAE 时

重点看：

| 文件 | 观察什么 |
|------|----------|
| `models/vae.py` | 模型输出和分类模型有什么不同（三元组 vs 单张量） |
| `trainer/vae.py` | 为什么不再用分类基类默认的损失和验证逻辑 |

### 当你学 DDPM 时

重点看：

| 文件 | 观察什么 |
|------|----------|
| `models/ddpm/unet.py` | 网络负责什么（预测噪声） |
| `models/ddpm/diffusion.py` | 调度器负责什么（加噪/去噪公式） |
| `trainer/diffusion.py` | trainer 为什么要把两者串起来 |

### 当你学 SR3 时

重点看：

| 文件 | 观察什么 |
|------|----------|
| `trainer/sr3.py` | 无条件扩散是怎么变成条件扩散的 |
| `models/sr3.py` | 为什么低清条件图会和当前状态图拼接到一起 |
| `models/ddpm/diffusion.py` | 为什么它是 DDPM 到 ResShift 的桥梁 |

### 当你学 ResShift 时

重点看：

| 文件 | 观察什么 |
|------|----------|
| `data/sr_dataset.py` | 数据样本从 `(image, label)` 变成 `(lr_up, hr)` |
| `trainer/resshift.py` | 训练目标从"一步回归"变成"多步残差恢复" |
| `models/resshift.py` | 调度器如何构造中间态和执行采样 |
| `utils/metrics.py` | 指标从分类准确率变成 PSNR |

---

## 常见问题

### 1. VAE 和普通 AutoEncoder 的根本区别是什么

普通 AutoEncoder 学的是一个确定性隐藏表示。

VAE 学的是一个潜变量分布，并且通过 KL 散度让这个分布更规整，方便采样生成。

| | AutoEncoder | VAE |
|---|-------------|-----|
| 编码器输出 | 确定向量 z | 分布参数 mu, logvar |
| 潜空间 | 不保证连续 | 被 KL 约束成连续平滑 |
| 生成能力 | 不能随机采样生成 | 可以从标准正态采样生成 |
| loss | 只有重构损失 | 重构损失 + KL 损失 |

### 2. DDPM 为什么不直接预测干净图，而要预测噪声

这是扩散模型里非常常见的训练形式。

从优化稳定性和理论推导上，预测噪声通常更自然，也更容易和扩散过程对应起来。

可以先记住一句话：

> 模型学会"噪声长什么样"，就能学会"怎么去掉噪声"。

从数学上看，预测噪声和预测干净图是等价的（可以互相转换），但预测噪声的训练信号更稳定。

### 3. SR3 和 ResShift 是一回事吗

不是。

两者都属于扩散超分，但侧重点不同：

- SR3 更像标准条件扩散（预测噪声，从纯噪声开始）
- ResShift 更强调残差和少步恢复（预测残差，从低清图附近开始）

### 4. ResShift 是不是就是 DDPM 换了个名字

不是。

它们都借用了扩散式、时间步式的建模思想，但关注点不同：

- DDPM 更像从噪声逐步生成
- ResShift 更像围绕条件图像做残差恢复

核心区别在于：建模对象不同（噪声 vs 残差）、采样起点不同（纯噪声 vs 低清图附近）、步数不同（500 vs 15）。

### 5. 为什么超分任务里要用 PSNR

因为超分是图像重建任务，通常关心生成结果和真值图像之间的像素误差。

PSNR 的计算公式：`PSNR = 10 * log10(data_range^2 / MSE)`

它不完美（和人类视觉感受不完全一致），但最适合作为起点。更高级的指标还有 SSIM、LPIPS 等。

### 6. 为什么 SR3 和 ResShift 的 UNet 结构几乎一样

因为"条件扩散"和"无条件扩散"在网络结构上的差异其实很小。关键变化在于：

- 训练逻辑不同（预测噪声 vs 预测残差）
- 调度器不同（`GaussianDiffusion` vs `ResidualShiftScheduler`）
- 采样过程不同（从纯噪声开始 vs 从低清图附近开始）

网络只需要把输入通道数从 1 改成 2（多了一个条件图通道），其他结构完全不变。

---

## 推荐顺序

1. 先理解 CNN 和通用 trainer
2. 再理解 VAE 的潜空间思想
3. 然后理解 DDPM 的时间步和噪声过程
4. 再看 SRResNet 的图像恢复思路
5. 然后看 SR3，理解条件扩散超分
6. 最后看 ResShift，理解更高效的扩散恢复

这个顺序的好处是：

- 从简单到复杂
- 从静态映射到动态过程
- 从无条件生成到条件恢复

---

## 可以怎么总结

如果只允许你用几句话总结这四类模型，可以这样讲：

- **VAE**：先把图像压到潜空间，再从潜空间解码回来。学的是分布，不是确定向量。
- **DDPM**：先把图像逐步加噪，再学会逐步去噪生成。学的是噪声长什么样。
- **SR3**：给定低清条件图，逐步生成更清晰的结果。是 DDPM 加上条件输入。
- **ResShift**：给定低质量条件图，逐步恢复它和目标图之间的残差。比 SR3 更高效，因为从低清图附近出发。

---

## 四种模型核心对比速查表

| | VAE | DDPM | SR3 | ResShift |
|---|-----|------|-----|----------|
| 任务类型 | 无条件生成 | 无条件生成 | 条件恢复 | 条件恢复 |
| 模型输入 | image | (x_t, t) | ([x_t, lr], t) | ([shifted, lr], t) |
| 训练目标 | 重构 + KL | 噪声预测 | 噪声预测 | 残差预测 |
| 采样方式 | z → 解码 | T步去噪 | T步条件去噪 | 少步残差恢复 |
| 采样步数 | 1 | 500-1000 | 50-100 | ~15 |
| 采样起点 | 随机 z | 纯噪声 | 纯噪声 | 低清图附近 |
| 条件信息 | 无 | 无 | 低清图 | 低清图 |
| 调度器 | 无 | GaussianDiffusion | GaussianDiffusion | ResidualShiftScheduler |
| 评估指标 | Loss ↓ | Noise Loss ↓ | PSNR ↑ | PSNR ↑ |
| 代码入口 | trainer/vae.py | trainer/diffusion.py | trainer/sr3.py | trainer/resshift.py |
