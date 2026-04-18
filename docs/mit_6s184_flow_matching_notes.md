# MIT 6.S184 流匹配与扩散模型课程对照笔记

这份文档基于 MIT 2026 年课程 `6.S184 Generative AI With Stochastic Differential Equations` 的讲义
`An Introduction to Flow Matching and Diffusion Models` 整理。

目标不是再写一份课程摘要，而是把三件事放在一起：

1. 课程每一章到底在讲什么
2. 这些知识点在本仓库里已经对应到哪些代码
3. 还有哪些知识点目前还没有落成实现

如果你手头有原始 PDF，可以配合这份文档一起读：

- `/home/dsw/Downloads/lecture_notes.pdf`

---

## 一眼看全：课程内容和仓库覆盖关系

| 课程主题 | 核心问题 | 仓库覆盖情况 | 主要代码 |
| --- | --- | --- | --- |
| 生成建模即采样 | 为什么生成任务可以看成从数据分布采样 | ✅ 已覆盖基础认知 | `docs/generative_basics.md`, `data/dataloader.py` |
| ODE / SDE 与生成 | 如何从噪声逐步走到数据 | ⚠️ 已覆盖离散时间扩散近似 | `models/ddpm/diffusion.py`, `trainer/diffusion.py` |
| Flow Matching | 如何直接学习噪声到数据的速度场 | ❌ 理论相关，未单独实现 | 当前可借 `SR3` / `ResShift` 作为过渡理解 |
| Score Matching | 为什么预测噪声等价于学习得分 | ⚠️ 已部分覆盖到 DDPM 训练逻辑 | `trainer/diffusion.py`, `models/ddpm/diffusion.py` |
| Guidance / CFG | 如何让生成结果服从条件 | ⚠️ 已覆盖"条件输入"，未实现 CFG | `models/sr3.py`, `trainer/sr3.py` |
| 大规模生成器 | UNet、VAE、latent space、DiT | ⚠️ 已覆盖 UNet、VAE；未覆盖 DiT/文本编码器 | `models/vae.py`, `models/ddpm/unet.py` |
| 离散扩散 | 如何把扩散思想迁移到文本/token | ❌ 未覆盖 | 无 |

覆盖状态说明：

- ✅ = 有完整实现和文档
- ⚠️ = 有部分实现，但缺少课程中的某些关键概念
- ❌ = 完全没有实现

---

## 1. 引言：生成模型为什么可以看成采样

### 课程在讲什么

课程一开始先统一建模视角：

- 图像、视频、分子结构都可以表示成高维向量
- 生成模型的目标不是"背答案"，而是从未知数据分布 `p_data` 中采样
- 条件生成则是在给定条件 `y` 的情况下采样 `p_data(x | y)`

### 数学直觉

想象你面前有一大堆手写数字图片。这些图片在高维空间中形成了一个"云团"——数据分布 `p_data`。

- **分类任务**：给你一张图，判断它在哪个类别 → 从图到标签的映射
- **生成任务**：从"云团"中随机抽一张新图 → 从分布中采样
- **条件生成**：只从"数字 7 那片云"中采样 → 条件分布采样

### 仓库代码对应

这部分和仓库最相关的地方，不在某个公式，而在任务定义方式：

| 任务类型 | 输入 | 输出 | 代码位置 |
|----------|------|------|----------|
| 分类 | `(image, label)` | 类别预测 | `trainer/classification.py` |
| 无条件生成 | `image` | 从 `p_data` 采样 | `trainer/diffusion.py`, `trainer/vae.py` |
| 条件恢复（超分） | `(lr, hr)` | 从 `p(hr \| lr)` 采样 | `trainer/sr3.py`, `trainer/resshift.py` |

关键代码走读——数据加载如何体现不同任务：

```python
# data/dataloader.py — 分类任务的数据
# 返回 (image, label)，训练目标是预测 label

# data/sr_dataset.py — 超分任务的数据
# 返回 (lr, hr)，训练目标是条件恢复
# lr = bicubic 下采样后的低清图
# hr = 原始高清图
```

### 建议理解重点

- 为什么分类和生成共用同一套训练骨架（`BaseTrainer`），但目标完全不同
- 为什么超分任务天然就是一种条件生成/条件恢复问题
- 为什么 VAE 和 DDPM 虽然都叫"生成模型"，但建模方式完全不同

---

## 2. Flow and Diffusion Models：生成是一个动态过程

### 课程在讲什么

课程第二章的核心是：

- 用 ODE 定义一个确定性的流：`dx/dt = v(x, t)`
- 用 SDE 在流上再加入噪声：`dx = v(x,t)dt + σdW`
- 通过数值方法逐步模拟从噪声到数据的演化

### 数学直觉

把生成过程想象成一条河：

- **ODE 视角**：水流从山顶（纯噪声）流向山脚（真实数据），每个位置的水流方向就是速度场 `v(x,t)`
- **SDE 视角**：水流中还有随机湍流 `σdW`，让路径不那么确定
- **DDPM 视角**：不关心连续水流，只在离散时间点上观察——每隔一段时间看一次水位

三者的关系：

```
连续时间 ODE  ──加噪声──→  连续时间 SDE  ──离散化──→  离散时间 DDPM
(确定性流)               (随机流)               (本仓库实现)
```

### 2.1 本仓库已经覆盖到什么

当前仓库主要覆盖的是扩散模型的**离散时间近似**，而不是完整连续时间 ODE/SDE 教学实现。

对应关系：

| 课程概念 | 仓库实现 | 代码位置 |
|----------|----------|----------|
| 正向加噪过程 | `GaussianDiffusion.q_sample()` | [diffusion.py:46-49](models/ddpm/diffusion.py#L46-L49) |
| 反向去噪一步 | `GaussianDiffusion.p_sample()` | [diffusion.py:66-69](models/ddpm/diffusion.py#L66-L69) |
| 从噪声预测原图 | `predict_x_start_from_noise()` | [diffusion.py:51-55](models/ddpm/diffusion.py#L51-L55) |
| 训练循环 | `DiffusionTrainer.train_one_epoch()` | [diffusion.py:62-90](trainer/diffusion.py#L62-L90) |
| 噪声预测 loss | `_compute_noise_prediction_loss()` | [diffusion.py:44-49](trainer/diffusion.py#L44-L49) |

### 2.2 关键代码走读：正向加噪

```python
# models/ddpm/diffusion.py — q_sample()
# 课程公式：x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε
def q_sample(self, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    return (
        self._extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
        self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
    )
```

**逐行解读**：

- `sqrt_alphas_cumprod` 就是课程里的 `√ᾱ_t`，表示信号保留比例
- `sqrt_one_minus_alphas_cumprod` 就是 `√(1-ᾱ_t)`，表示噪声强度
- `t` 越大 → `ᾱ_t` 越小 → 信号越弱、噪声越强
- `_extract()` 的作用是从预计算数组中取出当前时间步对应的值

### 2.3 关键代码走读：反向去噪

```python
# models/ddpm/diffusion.py — p_sample()
# 课程公式：x_{t-1} = μ_θ(x_t, t) + σ_t · z
def p_sample(self, model, x_t, t, clip_denoised_range=(-1.0, 1.0)):
    pred_noise = model(x_t, t)          # 网络预测噪声 ε_θ
    return self.p_sample_from_pred_noise(  # 用预测噪声计算 x_{t-1}
        x_t, t, pred_noise, clip_denoised_range=clip_denoised_range
    )
```

**关键观察**：

- 模型只负责预测噪声 `ε_θ(x_t, t)`
- 后验均值 `μ_θ` 由预测噪声和 `x_t` 线性组合得到
- `t=0` 时不加随机噪声（`nonzero_mask` 控制）

### 2.4 关键代码走读：训练循环

```python
# trainer/diffusion.py — _compute_noise_prediction_loss()
def _compute_noise_prediction_loss(self, images):
    batch_size = images.shape[0]
    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
    noise = torch.randn_like(images)
    x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
    predicted_noise = self.model(x_noisy, t)
    return self.criterion(predicted_noise, noise)  # MSE(ε_θ, ε)
```

**训练流程四步走**：

1. 随机采样时间步 `t`（让模型学会在所有噪声水平上预测）
2. 对原图加噪得到 `x_t`（正向过程）
3. 让网络根据 `(x_t, t)` 预测真实噪声 `ε`
4. 用 MSE 做监督（课程里的"噪声预测参数化"）

### 2.5 三层职责分离

先不要把它当成"会背 beta schedule 的公式"。更重要的是看清楚三层职责：

| 层 | 职责 | 代码 | 类比 |
|----|------|------|------|
| 网络 (`UNet`) | 在 `(x_t, t)` 条件下预测噪声 | `models/ddpm/unet.py` | 工人：按图纸干活 |
| 扩散过程 (`GaussianDiffusion`) | 时间步上的加噪和去噪公式 | `models/ddpm/diffusion.py` | 图纸：定义每步怎么做 |
| 训练器 (`DiffusionTrainer`) | 把两者接进训练循环 | `trainer/diffusion.py` | 工头：安排工人按图纸施工 |

### 2.6 当前还没有覆盖的部分

课程里强调的连续时间视角，目前仓库没有专门实现：

| 缺失内容 | 课程对应 | 难度 |
|----------|----------|------|
| ODE 求解器视角的 flow sampling | 第 2 章 | 中等 |
| Euler / Euler-Maruyama 的教学代码 | 第 2 章 | 简单 |
| Langevin dynamics | 第 4 章 | 中等 |
| 以 SDE 为主线的统一采样接口 | 第 2-4 章 | 较高 |

所以仓库现在更适合先建立 DDPM 直觉，而不是拿来完整复现课程第二章的全部数学形式。

---

## 3. Flow Matching：为什么它是这门课的核心

### 课程在讲什么

课程第三章是整份讲义最重要的部分之一。

核心思想不是"再造一个扩散模型"，而是换一种训练目标：

- 先设计一条从噪声到数据的概率路径 `p_t(x)`
- 再学习该路径上的向量场 `v_t(x)`
- 通过条件路径和边缘化技巧，把原本困难的整体学习问题转成简单的条件监督问题

### 数学直觉

**传统扩散模型（DDPM）**：

```
训练目标：预测噪声 ε
采样方式：从 x_T 开始，逐步去噪到 x_0
路径设计：由 β schedule 隐式定义
```

**流匹配（Flow Matching）**：

```
训练目标：预测速度场 v_t
采样方式：从 x_0（噪声）开始，沿速度场积分到 x_1（数据）
路径设计：显式定义插值路径
```

**打个比方**：

- DDPM 像"猜谜游戏"：给你一张模糊图，猜加了多少噪声
- Flow Matching 像"导航系统"：告诉你每个位置该往哪个方向走

### 三个关键概念

#### 概念 1：概率路径 `p_t(x)`

概率路径描述了从噪声分布 `p_0` 到数据分布 `p_1` 的连续过渡：

```
p_0(x) = N(0, I)           ← 纯高斯噪声
p_1(x) ≈ p_data(x)         ← 真实数据分布
p_t(x) = 中间的过渡分布
```

最简单的路径是线性插值：

```
x_t = (1-t) · x_0 + t · x_1,  t ∈ [0, 1]
```

其中 `x_0 ~ N(0, I)` 是噪声，`x_1` 是真实数据。

#### 概念 2：条件向量场 `u_t(x | x_1)`

向量场描述了概率路径上每个点"应该往哪个方向移动"：

```
u_t(x | x_1) = (x_1 - x_0) / (1 - 0) = x_1 - x_0    （线性插值时）
```

这个向量场是**条件**的——它依赖于目标数据点 `x_1`。

#### 概念 3：边缘化技巧

直接学习边缘向量场 `u_t(x)` 很困难（需要知道所有路径的叠加效果）。

但课程证明了一个关键结论：

> 对条件向量场做回归，等价于学习边缘向量场

也就是说，训练目标可以简化为：

```
L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - u_t(x_t | x_1)||² ]
```

其中 `v_θ` 是网络预测的速度场，`u_t(x_t | x_1)` 是条件向量场目标。

### 3.1 Flow Matching 和 DDPM 的本质区别

| | DDPM | Flow Matching |
|---|------|---------------|
| 训练目标 | 预测噪声 `ε` | 预测速度场 `v` |
| 采样方式 | 离散去噪步骤 | ODE 积分 |
| 路径定义 | 由 β schedule 隐式定义 | 显式定义插值路径 |
| 时间范围 | `t ∈ {0, 1, ..., T}` | `t ∈ [0, 1]` |
| 理论基础 | 变分推断 + score matching | 连续标准化流 (CNF) |
| 采样确定性 | 随机（有 σ·z 项） | 确定性（ODE） |

### 3.2 仓库中可以类比理解 Flow Matching 的代码

虽然仓库没有 Flow Matching 实现，但可以用现有代码建立直觉：

**类比 1：ResShift 的残差缩放 ≈ Flow Matching 的线性插值路径**

```python
# models/resshift.py — ResidualShiftScheduler.q_sample()
# ResShift: shifted = condition + scale * residual + noise_scale * noise
# Flow Matching: x_t = (1-t) * x_0 + t * x_1
```

两者都是在两个端点之间做插值，只是：
- ResShift 插值的是"低清图"和"高清图"之间的残差
- Flow Matching 插值的是"噪声"和"数据"之间的路径

**类比 2：DDPM 的噪声预测 ≈ Flow Matching 的速度场预测**

```python
# trainer/diffusion.py — 噪声预测
predicted_noise = self.model(x_noisy, t)  # 预测 ε

# Flow Matching 的速度场预测（概念代码，未实现）
# predicted_velocity = self.model(x_t, t)  # 预测 v_t
```

两者都是让网络学习一个"方向"：
- DDPM 学习"噪声的方向"
- Flow Matching 学习"数据流的方向"

**类比 3：DDPM 采样 ≈ Flow Matching ODE 采样**

```python
# DDPM 采样：逐步去噪
for step in reversed(range(self.timesteps)):
    img = self.diffusion.p_sample(self.model, img, t)

# Flow Matching ODE 采样（概念代码，未实现）
# for step in range(num_steps):
#     t = step / num_steps
#     v = model(x, t)           # 预测速度场
#     x = x + v * (1/num_steps) # Euler 步进
```

### 3.3 如果要把 Flow Matching 补进仓库，最小实现方案

```
新增文件：
├── models/flow_matching.py          # 概率路径 + ODE 采样器
├── trainer/flow_matching.py         # FlowMatchingTrainer
└── configs/generate/flow_matching.yaml
```

最小实现只需要四步：

**步骤 1：定义概率路径**

```python
# 最简单的高斯插值路径
class GaussianProbabilityPath:
    def __init__(self, sigma_min=0.001):
        self.sigma_min = sigma_min

    def sample(self, x1, t):
        # x1 是真实数据，t ∈ [0, 1]
        x0 = torch.randn_like(x1)  # 噪声
        mu_t = (1 - t) * x0 + t * x1  # 线性插值
        sigma_t = self.sigma_min      # 很小的噪声
        return mu_t + sigma_t * torch.randn_like(x1)

    def conditional_velocity(self, x0, x1, t):
        # 条件向量场目标
        return x1 - x0
```

**步骤 2：训练速度场**

```python
class FlowMatchingTrainer(BaseTrainer):
    def train_one_epoch(self, epoch):
        for x1, _ in self.train_loader:  # x1 是真实数据
            t = torch.rand(batch_size, 1, 1, 1, device=self.device)  # t ∈ [0, 1]
            x0 = torch.randn_like(x1)
            x_t = (1 - t) * x0 + t * x1  # 线性插值
            target_v = x1 - x0            # 条件速度场

            predicted_v = self.model(x_t, t.squeeze())  # 网络预测
            loss = F.mse_loss(predicted_v, target_v)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
```

**步骤 3：ODE 采样**

```python
@torch.no_grad()
def sample(self, num_steps=50):
    x = torch.randn(shape, device=self.device)  # 从噪声开始
    dt = 1.0 / num_steps
    for step in range(num_steps):
        t = torch.full((batch_size,), step / num_steps, device=self.device)
        v = self.model(x, t)      # 预测速度场
        x = x + v * dt            # Euler 步进
    return x
```

**步骤 4：配置文件**

```yaml
# configs/generate/flow_matching.yaml
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

### 3.4 为什么这部分仍然值得放进仓库笔记

因为这门课对应的工业背景，和仓库当前主线并不冲突。

你可以把当前仓库中的几条线看成它的前置知识：

```
DDPM → SR3 → ResShift → Flow Matching
 │       │       │          │
 │       │       │          └─ 更现代的生成范式
 │       │       └─ 恢复任务的高效扩散
 │       └─ 条件扩散
 └─ 基础扩散直觉
```

再往前走一步，才是课程里更现代的 flow matching 视角。

---

## 4. Score Functions and Score Matching：为什么 DDPM 在预测噪声

### 课程在讲什么

课程第四章把扩散模型放回更经典的概率视角：

- 得分函数是 `∇_x log p_t(x)`——概率密度函数的梯度
- 降噪得分匹配可以通过一个回归目标来训练
- 在常见高斯路径下，预测噪声、预测去噪结果、预测 score，本质上可以互相转换

### 数学直觉

**什么是得分函数？**

想象你站在一片雾中的山坡上，看不见全貌，但能感受到脚下的坡度。得分函数就是"坡度"——它告诉你"往哪个方向走，数据密度会增加"。

- 在数据密集的地方，得分函数指向密度更高的方向
- 在数据稀疏的地方，得分函数指向远离的方向

**为什么预测噪声等价于学习得分？**

课程给出了一个关键等式：

```
∇_x log p_t(x_t) = -ε_θ(x_t, t) / √(1 - ᾱ_t)
```

这意味着：

| 预测目标 | 等价于 | 关系 |
|----------|--------|------|
| 噪声 `ε` | 得分函数 `∇ log p_t` | 线性缩放 |
| 去噪结果 `x_0` | 得分函数 `∇ log p_t` | 非线性变换 |
| 速度场 `v` | 得分函数 `∇ log p_t` | 线性组合 |

**打个比方**：这就像温度可以用摄氏度、华氏度或开尔文来表示——数值不同，但描述的是同一个物理量。

### 4.1 仓库里的直接对应

这里和仓库最直接对应的是 `DiffusionTrainer` 的噪声回归训练：

```python
# trainer/diffusion.py — _compute_noise_prediction_loss()
def _compute_noise_prediction_loss(self, images):
    batch_size = images.shape[0]
    t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
    noise = torch.randn_like(images)
    x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
    predicted_noise = self.model(x_noisy, t)
    return self.criterion(predicted_noise, noise)  # MSE(ε_θ, ε)
```

**与 Score Matching 的等价关系**：

```
训练 loss = E[||ε_θ(x_t, t) - ε||²]           ← 仓库实现：噪声预测
          ∝ E[||s_θ(x_t, t) - ∇log p_t(x_t)||²]  ← 课程理论：得分匹配
```

虽然仓库没有单独写"score network"，但训练逻辑本身已经站在 score matching 的等价形式上。

### 4.2 三种参数化的代码对比

| 参数化 | 网络预测 | 从预测恢复 x_0 | 仓库实现 |
|--------|----------|----------------|----------|
| ε-预测 | `ε_θ(x_t, t)` | `x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t` | ✅ `predict_x_start_from_noise()` |
| x_0-预测 | `f_θ(x_t, t)` | 直接输出 | ❌ 未实现 |
| v-预测 | `v_θ(x_t, t)` | `x_0 = √ᾱ_t·x_t - √(1-ᾱ_t)·v` | ❌ 未实现 |

ε-预测的恢复代码：

```python
# models/ddpm/diffusion.py — predict_x_start_from_noise()
def predict_x_start_from_noise(self, x_t, t, pred_noise):
    return (
        self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
        self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise
    )
```

### 4.3 和课程相比少了什么

当前仓库没有专门把下面几件事拆开讲：

| 缺失内容 | 课程对应 | 重要性 |
|----------|----------|--------|
| score、noise、denoiser 三种参数化的线性关系 | 第 4 章 | 高——理解 DDPM 本质的关键 |
| Fokker-Planck 方程 | 第 4 章 | 中——连续时间理论 |
| SDE 采样下的概率流解释 | 第 4 章 | 中——统一 ODE/SDE 视角 |
| Langevin dynamics 采样 | 第 4 章 | 中——另一种采样方式 |

因此这部分如果只看代码，容易"会跑 DDPM，但不知道它和 score matching 的理论关系"。

这也是为什么这份文档需要把课程视角补回来。

---

## 5. Guidance：从无条件生成走向受控生成

### 课程在讲什么

课程第五章讨论的是如何在生成时加入条件，特别是：

- **Vanilla guidance**：直接把条件输入网络
- **Classifier guidance**：用额外分类器引导采样方向
- **Classifier-free guidance (CFG)**：同时训练有条件和无条件模型，推理时放大条件信号

### 数学直觉

**无条件生成**：模型只知道"数据长什么样"，随机采样

**条件生成（vanilla）**：模型知道"满足条件 y 的数据长什么样"

**CFG**：在推理时"放大"条件信号

```
CFG 公式：ε_θ_cfg(x_t, t, y) = ε_θ(x_t, t, ∅) + w · [ε_θ(x_t, t, y) - ε_θ(x_t, t, ∅)]
```

其中：
- `ε_θ(x_t, t, ∅)` 是无条件预测（"不知道条件时猜的噪声"）
- `ε_θ(x_t, t, y)` 是有条件预测（"知道条件时猜的噪声"）
- `w` 是引导强度（guidance scale），通常 `w > 1`
- 差值 `[有条件 - 无条件]` 就是"条件信号的方向"
- 乘以 `w` 就是"把条件信号放大"

**打个比方**：CFG 就像导航软件——无条件预测是"随便走"，有条件预测是"按路线走"，CFG 是"不仅按路线走，还加倍偏离随便走的方向"。

### 5.1 仓库里当前已经有的"条件生成"入口

仓库里已经有条件生成/条件恢复的雏形：

| 模型 | 条件输入方式 | 代码位置 |
|------|-------------|----------|
| DDPM | 无条件 | `models/ddpm/unet.py` → `in_channels=1` |
| SR3 | 通道拼接 `[x_t, lr]` | `models/sr3.py` → `in_channels=2` |
| ResShift | 通道拼接 `[x_t, lr]` | `models/resshift.py` → `in_channels=2` |

**条件注入的代码实现**：

```python
# models/sr3.py — SR3UNet
# 关键：in_channels=2，比无条件 UNet 多了一个通道
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)

# 训练时的条件注入
# trainer/sr3.py
predicted_noise = self.model(torch.cat([x_noisy, lr], dim=1), t)
#                                 ↑ 中间态    ↑ 条件图
#                                 通道拼接后送入网络
```

**最重要的桥梁认知**：

```
DDPM：模型看 (x_t, t)                    → 无条件
SR3： 模型看 ([x_t, lr], t)              → 条件（通道拼接）
CFG： 模型看 ([x_t, lr], t) 和 (x_t, t)  → 条件 + 无条件双分支
```

### 5.2 当前尚未实现的 CFG 内容

仓库没有实现课程第五章里更标准的条件生成组件：

| CFG 组件 | 课程对应 | 实现难度 |
|----------|----------|----------|
| 条件 dropout（训练时随机丢弃条件） | 第 5 章 | 简单 |
| 无条件/有条件双分支共享训练 | 第 5 章 | 中等 |
| CFG 推理公式 | 第 5 章 | 简单 |
| guidance scale 超参数 | 第 5 章 | 简单 |

**如果要在 SR3 上加 CFG，最小改动**：

```python
# 训练时：随机丢弃条件（10% 概率）
if random.random() < 0.1:
    lr_input = torch.zeros_like(lr)  # 用零图代替条件
else:
    lr_input = lr
predicted_noise = self.model(torch.cat([x_noisy, lr_input], dim=1), t)

# 推理时：CFG 公式
eps_uncond = self.model(torch.cat([x_noisy, zeros], dim=1), t)  # 无条件
eps_cond = self.model(torch.cat([x_noisy, lr], dim=1), t)       # 有条件
eps_cfg = eps_uncond + guidance_scale * (eps_cond - eps_uncond)  # 放大条件信号
```

---

## 6. Building Large-Scale Generators：从玩具实现到工业范式

### 课程在讲什么

课程第六章把理论推向工业实现，重点包括：

- 时间步嵌入（timestep embedding）
- 条件嵌入（condition embedding）
- UNet 架构
- DiT（Diffusion Transformer）
- VAE 与 latent diffusion

### 6.1 时间步嵌入：仓库已有实现

课程强调时间步信息必须以某种方式注入网络。仓库使用正弦位置编码：

```python
# models/ddpm/unet.py — timestep_embedding()
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    return embedding
```

**直觉**：这和 Transformer 的位置编码一样——用不同频率的正弦/余弦函数，让网络能区分不同的时间步。

**时间步嵌入如何注入 ResidualBlock**：

```python
# models/ddpm/unet.py — ResidualBlock
class ResidualBlock(TimestepBlock):
    def forward(self, x, t):
        h = self.conv1(x)
        h += self.time_emb(t)[:, :, None, None]  # 时间步信息加到特征图上
        h = self.conv2(h)
        return h + self.shortcut(x)
```

关键：时间步嵌入通过**逐元素相加**注入每个残差块，而不是拼接。

### 6.2 VAE：仓库已有明确实现

课程里强调 VAE 的作用，是先把高维图像压缩到低维潜空间，再在潜空间里做生成。

仓库已经有一条独立的 VAE 学习线：

| 组件 | 代码位置 | 职责 |
|------|----------|------|
| 编码器 | `models/vae.py` → `ConvVAE.encoder` | 图像 → `(mu, logvar)` |
| 重参数化 | `models/vae.py` → `ConvVAE.reparameterize()` | `(mu, logvar)` → 潜变量 `z` |
| 解码器 | `models/vae.py` → `ConvVAE.decoder` | 潜变量 `z` → 重建图像 |
| 训练 loss | `trainer/vae.py` → `vae_loss()` | 重构损失 + KL 散度 |

**关键代码走读**：

```python
# models/vae.py — 重参数化技巧
def reparameterize(self, mu, logvar):
    std = torch.exp(0.5 * logvar)   # 标准差 = exp(logvar / 2)
    eps = torch.randn_like(std)      # 从 N(0,1) 采样
    return mu + eps * std            # z = μ + σ·ε
```

为什么需要重参数化？因为直接从 `N(mu, std²)` 采样不可微，但 `z = mu + std * eps` 是可微的——梯度可以流过 `mu` 和 `std`。

```python
# trainer/vae.py — VAE loss
def vae_loss(x_recon, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")  # 重构项
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL 项
    return recon_loss + kl_loss
```

**两个 loss 项的直觉**：

| Loss 项 | 公式 | 直觉 | 作用 |
|---------|------|------|------|
| 重构损失 | `BCE(x_recon, x)` | 重建图要像原图 | 保证生成质量 |
| KL 散度 | `KL(q(z\|x) \|\| N(0,I))` | 潜空间要像标准正态 | 保证潜空间可采样 |

**KL 散度的展开**：

```
KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

- `μ²` 项：让均值靠近 0
- `σ²` 和 `log(σ²)` 项：让方差靠近 1
- 合在一起：让 `q(z|x)` 靠近 `N(0, I)`

### 6.3 UNet：仓库也已覆盖

课程里把 UNet 作为扩散模型的经典主干网络。

| 模型 | UNet 变体 | 条件方式 | 代码位置 |
|------|-----------|----------|----------|
| DDPM | `UNetModel` | 时间步嵌入 | `models/ddpm/unet.py` |
| SR3 | `SR3UNet(UNetModel)` | 时间步 + 通道拼接条件图 | `models/sr3.py` |
| ResShift | `ResShiftUNet(UNetModel)` | 时间步 + 通道拼接条件图 | `models/resshift.py` |

其中 `SR3UNet` 并没有换掉整个骨架，而是在已有扩散 UNet 上增加条件输入通道。
这点很适合教学，因为它把"无条件扩散"和"条件扩散"之间的结构关系展示得很直接。

```python
# models/sr3.py — SR3UNet 只改了 in_channels
class SR3UNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels, **kwargs)
```

**UNet 的关键组件**：

| 组件 | 代码位置 | 作用 |
|------|----------|------|
| `timestep_embedding()` | `unet.py:7-15` | 把时间步编码为向量 |
| `ResidualBlock` | `unet.py:46-66` | 带时间步注入的残差卷积块 |
| `AttentionBlock` | `unet.py:68+` | 自注意力机制 |
| 下采样 / 上采样 | `unet.py` 后续 | 特征图分辨率变化 |

### 6.4 当前未覆盖的工业级内容

课程这章里还有不少仓库尚未实现的现代组件：

| 组件 | 课程对应 | 工业应用 | 实现难度 |
|------|----------|----------|----------|
| DiT / Diffusion Transformer | 第 6 章 | SD3, Sora | 高 |
| AdaLN 条件注入 | 第 6 章 | DiT, SD3 | 中等 |
| CLIP / T5 文本编码器 | 第 6 章 | SD, DALL-E | 高 |
| Latent diffusion 训练链路 | 第 6 章 | Stable Diffusion | 中等 |
| 大规模图像/视频生成 pipeline | 第 6 章 | Sora, Movie Gen | 极高 |

**Latent Diffusion 的概念**（本仓库可以自然扩展的方向）：

```
当前仓库：
  图像空间 DDPM：直接在像素空间做扩散

Latent Diffusion：
  1. 用 VAE 编码器把图像压缩到潜空间
  2. 在潜空间做扩散（维度更低，训练更快）
  3. 用 VAE 解码器把潜变量解码回图像

代码路径：
  models/vae.py (已有) + models/ddpm/diffusion.py (已有)
  → 只需把 diffusion 的输入从图像换成 VAE 潜变量
```

---

## 7. Discrete Diffusion：从连续空间走向语言建模

### 课程在讲什么

课程第七章讨论离散扩散，也就是：

- 数据不再是欧几里得空间里的连续向量
- 而是 token、DNA 符号等离散序列
- 动态系统从 ODE/SDE 换成连续时间马尔可夫链 (CTMC)

### 数学直觉

**连续扩散（DDPM）**：

```
状态空间：R^d（连续向量）
转移方式：加高斯噪声
反向过程：去噪（预测噪声）
```

**离散扩散**：

```
状态空间：{1, 2, ..., K}^d（离散 token）
转移方式：按转移矩阵替换 token
反向过程：预测原始 token
```

**打个比方**：

- 连续扩散像"把一张照片逐渐加模糊"
- 离散扩散像"把一句话中的词逐渐替换成随机词"

### 仓库覆盖情况

目前仓库没有离散扩散的任何代码：

| 缺失内容 | 课程对应 | 实现难度 |
|----------|----------|----------|
| token 级别的数据管道 | 第 7 章 | 中等 |
| CTMC rate matrix | 第 7 章 | 高 |
| mask diffusion 语言模型 | 第 7 章 | 中等 |
| 交叉熵式离散流匹配训练器 | 第 7 章 | 高 |

如果未来仓库要扩成"从图像生成到语言扩散"的教学项目，这会是一个新的独立模块，而不是在现有 DDPM 文件上小修小补。

---

## 最推荐的阅读顺序

如果你的目标是借这门课反过来吃透本仓库，建议按下面顺序：

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
  本文档（MIT 课程对照笔记）
```

这样读的原因很简单：

- 先建立项目骨架
- 再建立 VAE 和 DDPM 的基础认知
- 再从 SR3 进入条件扩散
- 再理解课程里更现代的 flow matching 视角

如果一开始就直接啃流匹配公式，而没有 DDPM 和条件恢复直觉，读起来会非常抽象。

---

## 针对这门课，仓库接下来最值得补的内容

如果要让这个仓库更贴近 MIT 这份讲义，优先级建议如下：

| 优先级 | 内容 | 课程对应 | 实现难度 | 理由 |
|--------|------|----------|----------|------|
| 1 | 最小 `FlowMatchingTrainer` | 第 3 章 | 中等 | 课程核心，当前完全缺失 |
| 2 | ODE 采样脚本 + Euler solver | 第 3 章 | 简单 | Flow Matching 的推理方式 |
| 3 | 最小版 CFG 接口 | 第 5 章 | 简单 | 工业必备，SR3 上改动很小 |
| 4 | Latent diffusion 教学链路 | 第 6 章 | 中等 | VAE + Diffusion 已有，只需串联 |
| 5 | DiT 或离散扩散 | 第 6-7 章 | 高 | 前沿方向，但依赖较多 |

这个顺序比"先上大模型架构"更合理，因为它更符合这门课的知识递进，也更符合当前仓库的教学定位。

---

## 一句话总结

MIT 6.S184 这份讲义讲的是一条完整主线：

```
"生成就是采样" → ODE/SDE 驱动的 flow/diffusion → Flow Matching → Score Matching → Guidance → VAE/UNet/DiT/Latent Diffusion → 离散扩散
```

而本仓库当前最适合承接的，是这条主线中的教学骨架部分：

```
VAE → DDPM → SR3 条件扩散 → ResShift 风格恢复
```

它还没有完整覆盖讲义的最前沿部分，但已经具备了一个很好的承接基础。
