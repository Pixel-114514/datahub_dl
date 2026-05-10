# 生成模型知识补充

这份文档不追求把公式讲满，重点回答三个问题：

1. VAE、DDPM、SR3、ResShift 分别在解决什么问题
2. 它们的训练目标为什么不一样
3. 它们在本项目里分别对应哪些代码

> 如果你想知道"先学什么再学什么"，看 [学习路径](learning_path.md)。想了解项目为什么这样分层，看 [项目架构导读](architecture.md)。

---

## 先区分两类任务

分类是判别任务：输入一张图，输出它属于哪一类。目标是"做判断"。

VAE、DDPM、SR3、ResShift 是生成或恢复任务：要么从无到有生成一张图，要么从退化状态恢复出更好的图。目标是"建模数据分布"或"重建目标图像"。

`trainer/base.py` 里分类训练的核心就四行：

```python
logits = self.model(x)          # 前向
loss = self.criterion(logits, y) # 算 loss
loss.backward()                  # 反向传播
self.optimizer.step()            # 更新参数
```

生成模型也走这四步，但"模型输出什么"和"loss 算什么"完全不同。

---

## 为什么生成模型比分类难

分类只需要输出一个类别编号。生成模型要回答的问题更刁钻：什么样的像素组合才像真实图像？整体结构和局部细节怎么同时合理？有噪声或退化的话该恢复到什么程度？

所以生成模型会多出一些分类任务没有的概念：

- **潜变量 `z`**（VAE）：图像在低维空间的"压缩表示"
- **重参数化**（VAE）：让随机采样变得可导的技巧
- **时间步 `t`**（DDPM / SR3 / ResShift）：表示当前处于加噪/恢复过程的哪个阶段
- **噪声调度**（DDPM / SR3）：控制每一步加噪的强度
- **条件输入**（SR3 / ResShift）：给模型提供额外信息（如低清图）
- **采样过程**（DDPM / SR3 / ResShift）：推理时逐步生成/恢复图像

---

## VAE

VAE（Variational AutoEncoder，变分自编码器）做的事情很直观：编码器把图像压缩到低维潜空间，解码器再从潜空间把图像重建出来。

和普通 AutoEncoder 的区别：普通 AE 编码器直接输出一个隐藏向量，VAE 编码器输出一个分布——均值 `mu` 和方差参数 `logvar`，然后从这个分布里采样出 `z` 给解码器。

为什么要学分布？因为生成模型希望不只重建训练集里的样本，还能从潜空间随机采样生成新图。如果潜空间没有被约束成平滑连续的分布，随机采样出来的 `z` 大概率没有意义。

想象潜空间是一个地图。如果地图上只有几个孤立的"城市"（训练样本的位置），随机扔飞镖大概率落在荒野。KL 散度的作用就是把城市"铺开"，让整个地图都有意义。

**VAE 的 loss 有两项：**

- **重构损失**：解码后的图像接近原图
- **KL 损失**：潜变量分布接近标准正态

只有重构 loss 会退化成普通自编码器——能重建但不能生成。KL 太强又会让图像模糊——潜空间整齐但信息丢失。训练本质上是在两者之间找平衡。

```python
# trainer/vae.py
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss, recon_loss, kl_loss
```

**重参数化**解决的是一个工程问题：直接从 `N(mu, sigma^2)` 采样不可导，梯度传不回去。VAE 把它改写成 `z = mu + eps * sigma`（`eps ~ N(0,1)`），这样梯度就能通过 `mu` 和 `sigma` 传播。

```python
# models/vae.py
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std
```

VAE 生成的图像通常比扩散模型模糊，因为它优化的是"平均"输出。这是它的固有局限，不是实现的问题。

代码对应：

| 文件 | 看什么 |
|------|--------|
| [models/vae.py](../models/vae.py) | 编码器输出 `mu` 和 `logvar`；`reparameterize()` 把采样写成可训练形式 |
| [trainer/vae.py](../trainer/vae.py) | loss 返回三项（总 / 重构 / KL）；`_monitor_mode()` 返回 `"min"` |
| [configs/generate/vae.yaml](../configs/generate/vae.yaml) | `latent_dim: 20` 控制潜空间维度 |
| [inference_vae.py](../inference_vae.py) | 从潜空间随机采样并生成图像 |

> 想动手跑 VAE，看 [学习路径 > 第二阶段](learning_path.md#第二阶段vae-图片生成)。

---

## DDPM

DDPM（Denoising Diffusion Probabilistic Model）分两半理解：

**正向**：原图逐步加噪，经过 T 步变成纯噪声。

```
原图 x_0 → 加噪 x_1 → ... → 纯噪声 x_T
```

**反向**：训练一个网络，在每个时间步把噪声去掉一点。从纯噪声反复去噪，就能得到新图。

```
纯噪声 x_T → 去噪 x_{T-1} → ... → 生成图 x_0
```

### 时间步 t

不同时间步的噪声程度不同。模型必须知道当前面对的是轻微加噪的图（`t` 小，还比较清晰）还是几乎全是噪声的图（`t` 大）。所以模型输入是 `(x_t, t)`。

时间步通过正弦位置编码转成向量，注入 UNet 的每个残差块：

```python
# models/ddpm/unet.py
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
```

### 训练目标：预测噪声

最常见的做法是预测噪声：

1. 对原图加噪得到 `x_t`，同时保留这次加进去的噪声
2. 模型输入 `(x_t, t)`，输出对噪声的预测
3. 用 MSE 算预测噪声和真实噪声的差距

```python
# trainer/diffusion.py
def _compute_noise_prediction_loss(self, images):
    batch_size = images.shape[0]
    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
    noise = torch.randn_like(images)
    x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
    predicted_noise = self.model(x_noisy, t)
    return self.criterion(predicted_noise, noise)
```

如果模型能把噪声预测准，说明它学会了怎么把图像从噪声状态拉回真实分布。

从数学上看，预测噪声和预测干净图可以互相转换，但预测噪声的训练信号更稳定。

### 采样为什么慢

推理时从 `t=T` 一步步去噪到 `t=0`，每一步跑一次模型前向传播。T=500 就是 500 次。

DDPM 采样像雕刻——每次只去掉一小片碎屑，需要很多步。ResShift 更像修补——已经有了大致形状（低清图），只需要几步精修。

### beta schedule

`beta schedule` 控制每步加噪的强度。`linear` 从 0.0001 线性增长到 0.02（前期慢后期快），`cosine` 用余弦函数控制（更平滑）。

```python
# models/ddpm/diffusion.py
if beta_schedule == 'linear':
    betas = torch.linspace(0.0001, 0.02, timesteps)
elif beta_schedule == 'cosine':
    betas = cosine_beta_schedule(timesteps)
```

代码对应：

| 文件 | 看什么 |
|------|--------|
| [models/ddpm/unet.py](../models/ddpm/unet.py) | `timestep_embedding()`；`UNetModel.forward(x, timesteps)` |
| [models/ddpm/diffusion.py](../models/ddpm/diffusion.py) | `q_sample()` 加噪、`p_sample()` 去噪、`_extract()` 取系数 |
| [trainer/diffusion.py](../trainer/diffusion.py) | 训练时随机采样 `t`；验证时保存采样图片 |
| [configs/generate/ddpm.yaml](../configs/generate/ddpm.yaml) | `diffusion.timesteps: 500`、`diffusion.schedule: linear` |

> 想动手跑 DDPM，看 [学习路径 > 第三阶段](learning_path.md#第三阶段ddpm-基础扩散)。想深入 score matching 和 flow matching 的理论，看 [MIT 课程对照笔记](mit_6s184_flow_matching_notes.md)。

训练时不是从纯噪声走完整采样链——随机抽一个 `t`，只训练这个时间步的去噪能力。完整采样只在推理时做。

---

## SR3

SR3 是"把扩散模型引入超分"的桥梁。它说明了一件事：扩散模型不只能从纯噪声生成图像，也可以在条件图像的帮助下逐步恢复更高质量的结果。

### SR3 和 DDPM 的关系

SR3 不是推翻 DDPM，而是在 DDPM 基础上加了条件输入：

| | DDPM | SR3 |
|---|------|-----|
| 模型输入 | `(x_t, t)` | `([x_t, lr], t)` |
| 训练目标 | 预测噪声 | 预测噪声 |
| 采样起点 | 纯随机噪声 | 纯随机噪声 |
| 条件信息 | 无 | 低清图 `lr` |
| 网络结构 | UNet (in_channels=1) | UNet (in_channels=2) |

关键变化不是"时间步没了"，而是模型不只看当前噪声图，还看低清条件图。条件图通过通道拼接注入：

```python
# trainer/sr3.py
def _predict_noise(self, x_noisy, lr, t):
    return self.model(torch.cat([x_noisy, lr], dim=1), t)
```

`SR3UNet` 只做了一件事——把 `in_channels` 从 1 改成 2：

```python
# models/sr3.py
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

### SR3 在路径里的角色

SR3 是从 DDPM 到 ResShift 的桥梁。没有 SR3，直接从无条件扩散跳到 ResShift 跨度太大。有了 SR3，路径变成：

```
DDPM（无条件扩散）→ SR3（条件扩散）→ ResShift（少步 `x_0` 恢复）
```

每一步只变一个东西。

SR3 推理时从纯噪声开始，逐步去噪，每步都参考低清条件图。这和 ResShift 从低清图附近开始不同。

代码对应：

| 文件 | 看什么 |
|------|--------|
| [models/sr3.py](../models/sr3.py) | 整个类只有十几行，核心是 `in_channels=2` |
| [trainer/sr3.py](../trainer/sr3.py) | `torch.cat([x_noisy, lr], dim=1)` 是条件注入的关键 |
| [models/ddpm/diffusion.py](../models/ddpm/diffusion.py) | SR3 直接复用 DDPM 的 `GaussianDiffusion` |
| [configs/sr/sr3.yaml](../configs/sr/sr3.yaml) | `diffusion.timesteps: 50`（比 DDPM 的 500 少） |

> 想动手跑 SR3，看 [学习路径 > 第五阶段](learning_path.md#第五阶段sr3-条件扩散超分)。

---

## ResShift

ResShift 的出发点很直接：超分任务里低清图已经包含了大量结构信息，没必要像无条件生成那样从纯噪声开始慢慢采样。

### 为什么要关注 residual shifting

低清图里已经有大部分低频结构，因此 ResShift 不再像 DDPM / SR3 那样从纯噪声出发，而是把 forward / reverse 过程设计在 `x_0` 和 `y_0` 之间。这样最终先验更接近 LR 条件图，采样步数也可以明显缩短。

### 训练过程

```
1. 记 `x_0 = HR`，`y_0 = LR_up`
2. 随机采样时间步 t
3. 构造中间态：`x_t = x_0 + eta_t (y_0 - x_0) + kappa * sqrt(eta_t) * noise`
4. 模型预测 `x_0`：`predicted_x0 = model(cat([x_t, y_0], dim=1), t)`
5. `loss = MSE(predicted_x0, x_0)`
```

`eta_t` 随时间步增大。`t` 小时中间态接近 `x_0`，`t` 大时中间态接近 `y_0 + noise`。

```python
# models/resshift.py
def q_sample(self, target, condition, t, noise=None):
    if noise is None: noise = torch.randn_like(target)
    eta_t = self._extract(self.etas, t, target.shape)
    shifted = target + eta_t * (condition - target) + self.kappa * torch.sqrt(eta_t) * noise
    return shifted, target
```

### 推理过程

从低清图附近开始，逐步恢复 `x_0`：

```
1. current = LR_up + kappa * 随机噪声
2. for step = T-1, T-2, ..., 1:
3.     predicted_x0 = model(cat([current, LR_up], dim=1), step)
4.     current = posterior_mean(current, predicted_x0, step) + sqrt(var_t) * 噪声
5. current = predicted_x0  (step=0)
6. return current
```

最后一步直接输出 `predicted_x0`。

### ResShift vs DDPM vs SR3 vs SRResNet

| | DDPM | SR3 | ResShift | SRResNet |
|---|------|-----|----------|----------|
| 建模对象 | 噪声 | 噪声 | residual shifting | 残差 |
| 采样起点 | 纯噪声 | 纯噪声 | 低清图附近 | — |
| 采样步数 | 500-1000 | 50-100 | ~15 | 1 |
| 条件信息 | 无 | 低清图 | 低清图 | 低清图 |
| 训练目标 | 预测噪声 ε | 预测噪声 ε | 预测 `x_0` | 预测残差 R |

一句话总结：SRResNet 是"一步修图"，SR3 是"多步条件去噪"，ResShift 是"少步 `x_0` 恢复"。

代码对应：

| 文件 | 看什么 |
|------|--------|
| [models/resshift.py](../models/resshift.py) | `ResidualShiftScheduler.q_sample()` 构造中间态；`sample()` 逐步恢复 |
| [trainer/resshift.py](../trainer/resshift.py) | 训练时 `scheduler.q_sample(hr, lr, t)`；推理时 `scheduler.sample(model, lr)` |
| [configs/sr/resshift.yaml](../configs/sr/resshift.yaml) | `resshift.timesteps: 15`、`noise_level: 0.15`、`schedule: geometric` |

> ResShift 的算法详解、练习题和自查清单见 [ResShift 学习说明](resshift.md)。想动手跑 ResShift，看 [学习路径 > 第六阶段](learning_path.md#第六阶段resshift-少步扩散超分)。

---

## 四者演进关系

```
VAE（潜空间建模）
 ↓ 引入时间步
DDPM（逐步加噪/去噪）
 ↓ 引入条件输入
SR3（条件扩散超分）
 ↓ 围绕残差简化过程
ResShift（少步 `x_0` 恢复）
```

每一步只变一个核心概念。

VAE 教你潜变量、概率分布约束、重参数化。扩散模型教你时间步建模、逐步变换、训练目标可以是噪声而不是类别。两者代表了两种典型的生成建模思路：VAE 更适合讲"潜空间"，DDPM / ResShift 更适合讲"过程建模"。

---

## 和代码怎么对应

学 VAE 时重点看 [models/vae.py](../models/vae.py)（输出和分类模型有什么不同）和 [trainer/vae.py](../trainer/vae.py)（为什么不用基类的损失和验证逻辑）。

学 DDPM 时重点看 [models/ddpm/unet.py](../models/ddpm/unet.py)（预测噪声的网络）、[models/ddpm/diffusion.py](../models/ddpm/diffusion.py)（加噪/去噪公式）、[trainer/diffusion.py](../trainer/diffusion.py)（两者怎么串起来）。

学 SR3 时重点看 [trainer/sr3.py](../trainer/sr3.py)（无条件扩散怎么变成条件扩散）和 [models/sr3.py](../models/sr3.py)（为什么低清图和噪声图要拼接）。

学 ResShift 时重点看 [data/sr_dataset.py](../data/sr_dataset.py)（样本从 `(image, label)` 变成 `(lr_up, hr)`）、[trainer/resshift.py](../trainer/resshift.py)（训练目标从噪声预测变成 `x_0` 预测）、[models/resshift.py](../models/resshift.py)（调度器怎么构造中间态）。

---

## 常见问题

**VAE 和普通 AutoEncoder 的根本区别？** 普通 AE 学的是确定性隐藏表示，VAE 学的是潜变量分布。VAE 通过 KL 散度让潜空间连续平滑，可以随机采样生成。普通 AE 的潜空间不保证连续，没法做生成。

**DDPM 为什么不直接预测干净图而要预测噪声？** 从优化稳定性上，预测噪声更自然，也更容易和扩散过程对应。数学上两者等价（可以互相转换），但预测噪声的训练信号更稳定。

**SR3 和 ResShift 是一回事吗？** 不是。SR3 预测噪声，从纯噪声开始采样；ResShift 预测 `x_0`，从低清图附近开始采样。SR3 更像标准条件扩散，ResShift 更强调少步的 residual shifting 恢复。

**为什么超分用 PSNR？** 超分是重建任务，关心生成结果和真值的像素误差。PSNR = 10 * log10(data_range² / MSE)。它不完美（和人眼感受不完全一致），但作为起点够用。更高级的还有 SSIM、LPIPS。

**SR3 和 ResShift 的 UNet 为什么几乎一样？** 条件扩散和无条件扩散在网络结构上差异很小——只需把输入通道数从 1 改成 2。关键变化在训练逻辑和调度器，不在网络本身。
