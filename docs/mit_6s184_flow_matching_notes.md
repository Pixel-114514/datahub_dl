# MIT 6.S184 流匹配与扩散模型课程对照笔记

这份文档基于 MIT 2026 年课程 `6.S184 Generative AI With Stochastic Differential Equations` 的讲义整理。

不是再写一份课程摘要，而是把三件事放一起：

1. 课程每章在讲什么
2. 这些知识点在本仓库里已经对应到哪些代码
3. 还有哪些没有落成实现

> 前置知识：[学习路径](learning_path.md)、[生成模型知识补充](generative_basics.md)、[项目架构导读](architecture.md)。

如果有原始 PDF（`/home/dsw/Downloads/lecture_notes.pdf`），可以配合着看。

---

## 课程与仓库的覆盖关系

| 课程主题 | 核心问题 | 仓库覆盖情况 | 主要代码 |
| --- | --- | --- | --- |
| 生成建模即采样 | 为什么生成任务可以看成从分布采样 | ✅ 基础认知已有 | `docs/generative_basics.md` |
| ODE / SDE 与生成 | 如何从噪声逐步走到数据 | ⚠️ 已覆盖离散时间扩散近似 | `models/ddpm/diffusion.py` |
| Flow Matching | 如何直接学习噪声到数据的速度场 | ❌ 未单独实现 | 当前可借 SR3 / ResShift 过渡理解 |
| Score Matching | 预测噪声为什么等价于学习得分 | ⚠️ 部分覆盖到 DDPM 训练逻辑 | `trainer/diffusion.py` |
| Guidance / CFG | 如何让生成服从条件 | ⚠️ 已覆盖"条件输入"，未实现 CFG | `models/sr3.py` |
| 大规模生成器 | UNet、VAE、latent space、DiT | ⚠️ 已覆盖 UNet、VAE；未覆盖 DiT | `models/vae.py`, `models/ddpm/unet.py` |
| 离散扩散 | 把扩散迁移到文本/token | ❌ 未覆盖 | 无 |

---

## 1. 生成模型为什么可以看成采样

课程一开始的视角：图像、视频、分子结构都是高维向量，生成模型的目标不是"背答案"，而是从数据分布 `p_data` 中采样。条件生成是在给定条件 `y` 的情况下采样 `p_data(x | y)`。

想象一大堆手写数字图片在高维空间中形成一个"云团"——那就是 `p_data`。分类是从图到标签的映射，生成是从云团中随机抽一张新图，条件生成是只从"数字 7 那片云"里采样。

仓库里对应的部分：

- 分类：`(image, label)` → 预测 label
- 无条件生成：`image` → 从 `p_data` 采样
- 条件恢复（超分）：`(lr, hr)` → 从 `p(hr | lr)` 采样

关键认知：分类和生成共用同一套训练骨架（`BaseTrainer`），但目标完全不同。超分天然是条件生成/恢复问题。

---

## 2. Flow and Diffusion Models：生成是动态过程

课程第二章的核心：用 ODE 定义确定性流 `dx/dt = v(x, t)`，用 SDE 加入噪声 `dx = v(x,t)dt + σdW`，通过数值方法从噪声到数据逐步演化。

把生成过程想象成一条河：
- **ODE**：水流从山顶（噪声）流向山脚（数据），每个位置的水流方向是速度场 `v(x,t)`
- **SDE**：水流中有随机湍流 `σdW`
- **DDPM**：不关心连续水流，只在离散时间点上观察水位

```
连续时间 ODE  ──加噪声──→  连续时间 SDE  ──离散化──→  离散时间 DDPM
(确定性流)               (随机流)               (本仓库实现)
```

### 仓库已覆盖的部分

当前仓库覆盖的是扩散模型的**离散时间近似**：

| 课程概念 | 仓库实现 | 代码位置 |
|----------|----------|----------|
| 正向加噪 | `GaussianDiffusion.q_sample()` | `models/ddpm/diffusion.py` |
| 反向去噪一步 | `GaussianDiffusion.p_sample()` | `models/ddpm/diffusion.py` |
| 从噪声预测原图 | `predict_x_start_from_noise()` | `models/ddpm/diffusion.py` |
| 训练循环 | `DiffusionTrainer.train_one_epoch()` | `trainer/diffusion.py` |
| 噪声预测 loss | `_compute_noise_prediction_loss()` | `trainer/diffusion.py` |

### 正向加噪代码走读

```python
# models/ddpm/diffusion.py — q_sample()
# 公式：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
def q_sample(self, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
```

`sqrt_alphas_cumprod` 是 `√ᾱ_t`（信号保留比例），`sqrt_one_minus_alphas_cumprod` 是 `√(1-ᾱ_t)`（噪声强度）。`t` 越大信号越弱噪声越强。`_extract()` 从预计算数组中取出当前时间步对应的值。

### 反向去噪代码走读

```python
# models/ddpm/diffusion.py — p_sample()
def p_sample(self, model, x_t, t, clip_denoised_range=(-1.0, 1.0)):
    pred_noise = model(x_t, t)          # 网络预测噪声
    return self.p_sample_from_pred_noise(
        x_t, t, pred_noise, clip_denoised_range=clip_denoised_range
    )
```

模型只负责预测噪声 `ε_θ(x_t, t)`，后验均值由预测噪声和 `x_t` 线性组合得到。`t=0` 时不加随机噪声。

### 训练循环代码走读

```python
# trainer/diffusion.py
def _compute_noise_prediction_loss(self, images):
    batch_size = images.shape[0]
    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
    noise = torch.randn_like(images)
    x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
    predicted_noise = self.model(x_noisy, t)
    return self.criterion(predicted_noise, noise)  # MSE(ε_θ, ε)
```

四步：随机采样 `t` → 对原图加噪 → 网络预测噪声 → MSE 监督。

### 三层职责分离

| 层 | 职责 | 代码 | 类比 |
|----|------|------|------|
| 网络 (`UNet`) | 预测噪声 | `models/ddpm/unet.py` | 工人 |
| 扩散过程 (`GaussianDiffusion`) | 加噪/去噪公式 | `models/ddpm/diffusion.py` | 图纸 |
| 训练器 (`DiffusionTrainer`) | 接进训练循环 | `trainer/diffusion.py` | 工头 |

### 还没覆盖的部分

| 缺失内容 | 课程对应 | 难度 |
|----------|----------|------|
| ODE 求解器视角的 flow sampling | 第 2 章 | 中等 |
| Euler / Euler-Maruyama 教学代码 | 第 2 章 | 简单 |
| Langevin dynamics | 第 4 章 | 中等 |
| SDE 为主线的统一采样接口 | 第 2-4 章 | 较高 |

仓库现在更适合先建立 DDPM 直觉，不是拿来完整复现课程第二章的全部数学。

---

## 3. Flow Matching：课程核心

课程第三章是最重要的部分之一。核心不是"再造一个扩散模型"，而是换一种训练目标：

- 先设计从噪声到数据的概率路径 `p_t(x)`
- 再学习路径上的向量场 `v_t(x)`
- 通过条件路径和边缘化技巧，把整体学习转成简单的条件监督

### DDPM vs Flow Matching

```
DDPM：         预测噪声 ε，从 x_T 逐步去噪到 x_0，路径由 β schedule 隐式定义
Flow Matching：预测速度场 v，从 x_0（噪声）沿速度场积分到 x_1（数据），路径显式定义
```

DDPM 像"猜谜游戏"：给你模糊图，猜加了多少噪声。Flow Matching 像"导航系统"：告诉你每个位置该往哪走。

### 三个关键概念

**概率路径 `p_t(x)`**：从噪声分布 `p_0 = N(0,I)` 到数据分布 `p_1 ≈ p_data` 的连续过渡。最简单的路径是线性插值 `x_t = (1-t) · x_0 + t · x_1`。

**条件向量场 `u_t(x | x_1)`**：描述路径上每个点该往哪移动。线性插值时 `u_t = x_1 - x_0`。

**边缘化技巧**：直接学边缘向量场 `u_t(x)` 很困难，但对条件向量场做回归等价于学边缘向量场。训练目标：

```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - u_t(x_t | x_1)||² ]
```

### 和 DDPM 的本质区别

| | DDPM | Flow Matching |
|---|------|---------------|
| 训练目标 | 预测噪声 ε | 预测速度场 v |
| 采样方式 | 离散去噪步骤 | ODE 积分 |
| 路径定义 | β schedule 隐式定义 | 显式定义插值路径 |
| 时间范围 | `t ∈ {0,...,T}` | `t ∈ [0, 1]` |
| 采样确定性 | 随机（有 σ·z） | 确定性（ODE） |

### 仓库中可以类比的代码

虽然仓库没有 Flow Matching 实现，但可以用现有代码建立直觉：

**ResShift 的残差缩放 ≈ Flow Matching 的线性插值路径**

```python
# ResShift: shifted = condition + scale * residual + noise_scale * noise
# Flow Matching: x_t = (1-t) * x_0 + t * x_1
```

两者都在两个端点之间做插值。ResShift 插值的是低清图和高清图之间的残差，Flow Matching 插值的是噪声和数据之间的路径。

**DDPM 的噪声预测 ≈ Flow Matching 的速度场预测**

```python
# DDPM: predicted_noise = self.model(x_noisy, t)  # 预测 ε
# Flow Matching: predicted_velocity = self.model(x_t, t)  # 预测 v_t
```

都是让网络学一个"方向"。DDPM 学噪声的方向，Flow Matching 学数据流的方向。

### 如果要补进仓库

最小实现四步：

**1. 定义概率路径**

```python
class GaussianProbabilityPath:
    def sample(self, x1, t):
        x0 = torch.randn_like(x1)
        mu_t = (1 - t) * x0 + t * x1
        return mu_t + 0.001 * torch.randn_like(x1)

    def conditional_velocity(self, x0, x1, t):
        return x1 - x0
```

**2. 训练速度场**

```python
class FlowMatchingTrainer(BaseTrainer):
    def train_one_epoch(self, epoch):
        for x1, _ in self.train_loader:
            t = torch.rand(batch_size, 1, 1, 1, device=self.device)
            x0 = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1
            target_v = x1 - x0
            predicted_v = self.model(x_t, t.squeeze())
            loss = F.mse_loss(predicted_v, target_v)
            # ... backward + step
```

**3. ODE 采样**

```python
@torch.no_grad()
def sample(self, num_steps=50):
    x = torch.randn(shape, device=self.device)
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size,), step / num_steps, device=self.device)
        v = self.model(x, t)
        x = x + v * dt  # Euler 步进
    return x
```

**4. 配置文件**

```yaml
model:
  type: unet
  params:
    in_channels: 1
    out_channels: 1
flow_matching:
  path: gaussian
  sigma_min: 0.001
  num_sample_steps: 50
```

### 为什么值得放进笔记

当前仓库的知识线可以看成 Flow Matching 的前置：

```
DDPM → SR3 → ResShift → Flow Matching
```

先建立 DDPM 直觉，再理解条件恢复，最后才是课程里更现代的 flow matching 视角。直接啃流匹配公式而没有 DDPM 基础，会非常抽象。

---

## 4. Score Functions：为什么 DDPM 在预测噪声

课程第四章把扩散放回概率视角：得分函数是 `∇_x log p_t(x)`——概率密度的梯度。

想象站在雾中的山坡上，看不见全貌但能感受到脚下坡度。得分函数就是"坡度"——告诉你往哪走数据密度会增加。

课程给出的关键等式：

```
∇_x log p_t(x_t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
```

这意味着预测噪声 ε 和学得分函数 `∇ log p_t` 是线性缩放关系，本质等价。

| 预测目标 | 等价于 | 关系 |
|----------|--------|------|
| 噪声 ε | 得分函数 | 线性缩放 |
| 去噪结果 x_0 | 得分函数 | 非线性变换 |
| 速度场 v | 得分函数 | 线性组合 |

就像温度可以用摄氏度、华氏度或开尔文表示——数值不同，描述的是同一个物理量。

仓库里的直接对应：

```python
# trainer/diffusion.py
# 训练 loss = E[||ε_θ(x_t, t) - ε||²]     ← 仓库：噪声预测
#           ∝ E[||s_θ - ∇log p_t||²]        ← 课程：得分匹配
```

仓库没有单独写"score network"，但训练逻辑已经站在 score matching 的等价形式上。

### 三种参数化

| 参数化 | 网络预测 | 从预测恢复 x_0 | 仓库 |
|--------|----------|----------------|------|
| ε-预测 | `ε_θ(x_t, t)` | `(x_t - √(1-ᾱ_t)·ε) / √ᾱ_t` | ✅ |
| x_0-预测 | `f_θ(x_t, t)` | 直接输出 | ❌ |
| v-预测 | `v_θ(x_t, t)` | `√ᾱ_t·x_t - √(1-ᾱ_t)·v` | ❌ |

### 还没覆盖的

| 缺失内容 | 重要性 |
|----------|--------|
| score、noise、denoiser 三种参数化的线性关系 | 高——理解 DDPM 本质的关键 |
| Fokker-Planck 方程 | 中——连续时间理论 |
| SDE 采样下的概率流解释 | 中——统一 ODE/SDE 视角 |
| Langevin dynamics 采样 | 中——另一种采样方式 |

只看代码容易"会跑 DDPM，不知道它和 score matching 的关系"。

---

## 5. Guidance：从无条件到受控生成

课程第五章：如何在生成时加入条件。

- **Vanilla guidance**：直接把条件输入网络
- **Classifier guidance**：用额外分类器引导采样
- **Classifier-free guidance (CFG)**：同时训练有条件和无条件模型，推理时放大条件信号

CFG 公式：

```
ε_cfg(x_t, t, y) = ε(x_t, t, ∅) + w · [ε(x_t, t, y) - ε(x_t, t, ∅)]
```

`ε(x_t, t, ∅)` 是无条件预测（"不知道条件时猜的"），`ε(x_t, t, y)` 是有条件预测，`w` 是引导强度。差值 `[有条件 - 无条件]` 是条件信号的方向，乘以 `w` 就是放大这个方向。

CFG 像导航：无条件预测是"随便走"，有条件预测是"按路线走"，CFG 是"不仅按路线走，还加倍偏离随便走的方向"。

仓库里已有的条件生成入口：

| 模型 | 条件注入方式 |
|------|-------------|
| DDPM | 无条件，`in_channels=1` |
| SR3 | 通道拼接 `[x_t, lr]`，`in_channels=2` |
| ResShift | 通道拼接 `[x_t, lr]`，`in_channels=2` |

如果要在 SR3 上加 CFG，最小改动：

```python
# 训练时：10% 概率丢弃条件
if random.random() < 0.1:
    lr_input = torch.zeros_like(lr)
else:
    lr_input = lr
predicted_noise = self.model(torch.cat([x_noisy, lr_input], dim=1), t)

# 推理时：CFG 公式
eps_uncond = self.model(torch.cat([x_noisy, zeros], dim=1), t)
eps_cond = self.model(torch.cat([x_noisy, lr], dim=1), t)
eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
```

---

## 6. 大规模生成器：从玩具到工业

课程第六章把理论推向工业实现：时间步嵌入、条件嵌入、UNet、DiT、VAE 与 latent diffusion。

### 时间步嵌入（仓库已有）

正弦位置编码，和 Transformer 的位置编码一样：

```python
# models/ddpm/unet.py
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half) / half)
    args = timesteps[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
```

时间步嵌入通过逐元素相加注入每个残差块，不是拼接：

```python
class ResidualBlock(TimestepBlock):
    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]  # 时间步加到特征图上
        h = self.conv2(h)
        return h + self.shortcut(x)
```

### VAE（仓库已有）

编码器把图像压到潜空间，解码器从潜变量恢复图像。重参数化 `z = mu + std * eps` 让采样可微。

两个 loss 项：重构损失管"重建得像不像"，KL 散度管"潜空间整不整齐"。

### UNet（仓库已有）

| 模型 | UNet 变体 | 条件方式 |
|------|-----------|----------|
| DDPM | `UNetModel` | 时间步嵌入 |
| SR3 | `SR3UNet(UNetModel)` | 时间步 + 通道拼接 |
| ResShift | `ResShiftUNet(UNetModel)` | 时间步 + 通道拼接 |

`SR3UNet` 只在已有扩散 UNet 上增加条件输入通道，没有换骨架。这把无条件和条件扩散之间的关系展示得很直接。

### 还没覆盖的工业级内容

| 组件 | 工业应用 | 实现难度 |
|------|----------|----------|
| DiT / Diffusion Transformer | SD3, Sora | 高 |
| AdaLN 条件注入 | DiT, SD3 | 中等 |
| CLIP / T5 文本编码器 | SD, DALL-E | 高 |
| Latent diffusion 训练链路 | Stable Diffusion | 中等 |

**Latent Diffusion 的概念**（仓库可以自然扩展的方向）：

当前仓库在像素空间做扩散。Latent Diffusion 先用 VAE 编码器把图像压到潜空间，在潜空间做扩散（维度更低，训练更快），再用解码器解码回来。代码路径上 `models/vae.py`（已有）+ `models/ddpm/diffusion.py`（已有）只需把 diffusion 的输入从图像换成 VAE 潜变量。

---

## 7. 离散扩散：从连续到语言

课程第七章：数据不再是连续向量，而是 token、DNA 符号等离散序列。动态系统从 ODE/SDE 换成连续时间马尔可夫链 (CTMC)。

连续扩散像"把照片逐渐加模糊"，离散扩散像"把一句话中的词逐渐替换成随机词"。

仓库目前没有离散扩散的任何代码。如果未来要扩到"从图像生成到语言扩散"的教学，这会是一个新的独立模块。

---

## 推荐阅读顺序

```
第 1 步：项目骨架
  docs/architecture.md → docs/learning_path.md

第 2 步：基础生成模型
  docs/generative_basics.md → models/vae.py + trainer/vae.py

第 3 步：扩散模型
  models/ddpm/diffusion.py + trainer/diffusion.py

第 4 步：条件扩散
  models/sr3.py + trainer/sr3.py

第 5 步：高效恢复扩散
  trainer/resshift.py + docs/resshift.md

第 6 步：课程理论对照
  本文档
```

先建立项目骨架，再建立 VAE 和 DDPM 基础认知，再从 SR3 进入条件扩散，最后看课程里更现代的 flow matching 视角。直接啃流匹配公式而没有 DDPM 和条件恢复直觉，会非常抽象。

---

## 仓库接下来最值得补的内容

| 优先级 | 内容 | 课程对应 | 实现难度 | 理由 |
|--------|------|----------|----------|------|
| 1 | 最小 FlowMatchingTrainer | 第 3 章 | 中等 | 课程核心，完全缺失 |
| 2 | ODE 采样脚本 + Euler solver | 第 3 章 | 简单 | Flow Matching 的推理方式 |
| 3 | 最小版 CFG 接口 | 第 5 章 | 简单 | 工业必备，SR3 上改动很小 |
| 4 | Latent diffusion 教学链路 | 第 6 章 | 中等 | VAE + Diffusion 已有，只需串联 |
| 5 | DiT 或离散扩散 | 第 6-7 章 | 高 | 前沿方向，依赖较多 |

这个顺序比"先上大模型架构"更合理，更符合课程的知识递进，也更符合仓库的教学定位。

---

## 一句话总结

MIT 6.S184 讲的主线：

```
"生成就是采样" → ODE/SDE → Flow Matching → Score Matching → Guidance → VAE/UNet/DiT/Latent Diffusion → 离散扩散
```

本仓库当前最适合承接的是教学骨架部分：

```
VAE → DDPM → SR3 条件扩散 → ResShift 风格恢复
```

还没完整覆盖讲义最前沿，但已经具备了很好的承接基础。
