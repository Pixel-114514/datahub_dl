# 学习路径

这份文档讲的是整条学习路径该怎么走，不是某个模型的细节。

从手写数字识别开始，一路走到扩散超分。每一步只变一个东西，保证你能跟上。

推荐顺序：

1. 手写数字识别
2. VAE 图片生成
3. DDPM 基础扩散
4. 超分基线
5. SR3 条件扩散超分
6. ResShift 少步扩散超分
7. 工程实践与扩展练习

> **训练时间**：每个阶段的预估时间标注在各阶段末尾。想快速验证流程的话，在配置里加 `data.max_train_samples: 512` 和 `data.max_val_samples: 128`，不影响对代码的理解。

---

## 第一阶段：手写数字识别

先把一个深度学习项目完整跑通。不追求高分，追求看懂每一步在干什么。

跑起来：

```bash
python main.py --config configs/classification/cnn.yaml
# 没有 GPU 的话加 --device cpu
```

你应该会看到 loss 逐步下降，验证准确率逐步上升。类似这样：

```
Epoch [1] Train Loss: 0.4521
Epoch [1] Val Acc: 0.9210
...
Epoch [10] Train Loss: 0.0132
Epoch [10] Val Acc: 0.9912
Training finished. Best Val Acc: 0.9921 @ epoch 8
```

> 首次运行会自动下载 MNIST 到 `./data`，约 11MB。GPU 上 1-2 分钟跑完，CPU 上 3-5 分钟。

这个阶段要搞清楚的几件事：

- **样本和标签**：MNIST 里每张图是一个样本，标签是 0-9 的整数。`data/dataloader.py` 返回的每个 batch 就是一组 `(image, label)`。
- **前向传播和 loss**：数据从输入层算到输出，得到预测值；预测值和真实标签的差距就是 loss。loss 越小说明模型越准。
- **反向传播和优化器**：`loss.backward()` 算梯度，`optimizer.step()` 更新参数。这两行是 PyTorch 训练的核心。
- **训练集和验证集**：训练集用来学，验证集用来检查有没有"死记硬背"。

对应代码不多，建议按这个顺序看：

1. `configs/classification/cnn.yaml` — 看配置长什么样，每个字段控制什么
2. `main.py` — 看配置怎么被读取、trainer 怎么被选出来
3. `trainer/base.py` — 看 `train_one_epoch()` 和 `evaluate()` 的完整流程
4. `models/cnn.py` — 看 `forward()` 的输入输出形状

跑完 CNN 后再跑一下 ResNet：

```bash
python main.py --config configs/classification/resnet.yaml
```

对比两者：CNN 参数量约 7 万，ResNet 更大。关键区别是 ResNet 有残差连接（`x + self.block(x)`）。在 `models/cnn.py` 和 `models/resnet.py` 里可以直接对比。

想观察超参的影响，可以试试：

- `batch_size` 从 128 改成 32 — 训练变慢，但梯度估计更准
- `lr` 从 0.001 改成 0.01 — 可能更快收敛，也可能震荡
- `epochs` 从 10 改成 3 — 看看欠拟合长什么样

**几个容易困惑的地方：**

验证准确率比训练低是正常的，模型在训练集上"见过"，验证集上没见过。差距太大（比如 99% vs 85%）才说明过拟合。

每次跑结果可能不完全一样 — `seed: 42` 保证同设备同版本下可复现，但换设备或 PyTorch 版本会有浮点差异。

---

## 第二阶段：VAE 图片生成

从这一步开始，视角从"判别"转到"生成"。

分类关心的是"这张图是什么类别"，VAE 关心的是"怎么把图像压缩到低维空间，再从低维空间恢复出来"。

跑起来：

```bash
python main.py --config configs/generate/vae.yaml
# 跑完后生成图像：
python inference_vae.py
```

> VAE 的 loss 是越小越好（不是越大越好）。GPU 上 3-5 分钟，CPU 上 10-15 分钟。

VAE 的核心就三步：编码器把图像压成 `mu` 和 `logvar`（潜变量分布的均值和方差），重参数化采样出 `z`，解码器从 `z` 重建图像。

和普通 AutoEncoder 的区别在于：普通 AE 学的是一个确定的隐藏向量，VAE 学的是一个分布。通过 KL 散度把这个分布约束成接近标准正态，这样从潜空间随机采样才不会采到没意义的点。

**重构损失**管"重建得像不像"，**KL 损失**管"潜空间整不整齐"。两个 loss 互相拉扯——KL 太强图像会模糊，KL 太弱潜空间没法采样。

训练时你会看到三个数字：

```
Epoch [1] Train Loss: 258.1234 | Recon: 254.5678 | KL: 3.5556
```

初期 KL 很小（潜空间还没被整理），后面 KL 会逐渐增大。

值得动手试的事：

- 把 `latent_dim` 从 20 改成 2 — 维度太低信息不够，生成质量会明显下降
- 跑完后用 `inference_vae.py` 看看从潜空间采样出来的图长什么样
- 对比 VAE 生成和后续 DDPM 生成的质量差异

代码重点看四个文件：

- `models/vae.py` — 编码器输出 `mu` 和 `logvar`，`reparameterize()` 做重参数化
- `trainer/vae.py` — loss 为什么返回三项（总 loss / 重构 / KL），`_monitor_mode()` 为什么返回 `"min"`
- `configs/generate/vae.yaml` — `latent_dim: 20` 控制潜空间维度
- `inference_vae.py` — 如何从潜空间随机采样并生成

**VAE 生成模糊是正常的**——它优化的是"平均"输出，均值天然是模糊的。后面的扩散模型在生成质量上会好很多。

---

## 第三阶段：DDPM 基础扩散

VAE 是"一步到位"：编码 → 采样 → 解码。

DDPM 是"多步迭代"：原图逐步加噪变成纯噪声（正向），纯噪声逐步去噪恢复出图像（反向）。

```bash
python main.py --config configs/generate/ddpm.yaml
```

> DDPM 默认只设了 1 个 epoch，建议改到 30-50 才能看到像样的生成效果。首次运行采样会比较慢（500 步去噪）。GPU 上 1 epoch 约 2-3 分钟，CPU 上 10-20 分钟。

每个 epoch 结束后，训练器会自动保存一张采样图片到实验目录。早期全是噪声，后面会逐渐成形。

DDPM 的关键概念：

- **时间步 `t`**：表示当前加噪程度。`t=0` 是原图，`t=T` 是纯噪声。
- **`q_sample`**：给原图加噪，公式 `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`
- **`p_sample`**：给定 `x_t` 和预测噪声，推算 `x_{t-1}`
- **噪声预测**：模型的目标不是预测图像，而是预测"加了什么噪声"

训练时并不是从纯噪声走完整个采样链。它是随机抽一个时间步 `t`，只训练模型在这个 `t` 上的去噪能力。完整采样只在推理时才做。这比走完整链路高效得多，而且数学上等价。

网络结构用的是 UNet（`models/ddpm/unet.py`），有下采样-瓶颈-上采样的对称结构。时间步信息通过正弦位置编码注入每个残差块。

值得动手试的事：

- 先跑 1 epoch 确认流程没问题，再把 epochs 调大
- 把 `timesteps` 从 500 改成 100 — 训练更快，但生成质量可能下降
- 把 `schedule` 从 `linear` 改成 `cosine` — 加噪过程更平滑

**采样慢的原因**：推理时要从 `t=T` 一步步去噪到 `t=0`，每一步都要跑一次模型前向传播。T=500 就是 500 次。这就是后面 ResShift 要解决的问题。

扩散模型通常把像素从 [0,1] 映射到 [-1,1]（配置里 `value_range: minus_one_one`），这样加噪后值域对称，训练更稳定。

---

## 第四阶段：超分基线

前面三步分别是分类和生成，现在进入"图像恢复"——已有低质量图，目标是恢复出高质量图。

```bash
python main.py --config configs/sr/srresnet.yaml
```

超分的数据不是直接拿分类数据来用。`data/sr_dataset.py` 会把原始图像包装一下：下采样得到低清图 `lr`，再插值回原尺寸得到 `lr_up`，最后返回 `(lr_up, hr)` 这对训练样本。

这样做的好处是不需要准备 DIV2K 那种大数据集，用 MNIST 就能跑通整个流程。

评估指标从准确率换成了 PSNR（峰值信噪比），衡量恢复图和真值图的像素级差异，越高越好。MNIST 上 2x 超分，SRResNet 通常能到 22-25 dB。

SRResNet 的思路是残差学习：网络不直接输出超分图，而是输出"残差"（高频细节），最终结果 = 输入 + 残差。低清图已经包含了大部分低频信息，网络只需要学"补细节"。

代码重点看：

- `data/sr_dataset.py` — `(image, label)` 怎么变成 `(lr_up, hr)`
- `models/sr.py` — 残差块结构、全局跳跃连接 `x + residual`
- `trainer/sr.py` — `BaseSRTrainer` 是 SR3 和 ResShift 的父类
- `utils/metrics.py` — PSNR 的计算

可以试的事：

- 修改 `scale_factor` 从 2 改成 4 — 超分难度增加，PSNR 会下降
- 修改 `num_blocks` 从 6 改成 3 — 更浅的网络表现如何
- 设置 `noise_std: 0.01` — 让低清图带噪声，看恢复效果

---

## 第五阶段：SR3 条件扩散超分

这一步是从 DDPM 到 ResShift 之间最重要的桥梁。

DDPM 是无条件扩散：从纯噪声生成图像。SR3 加了一个变化——把低清条件图也喂给模型。模型输入从 `(x_t, t)` 变成了 `([x_t, lr], t)`，仅此而已。

```bash
python main.py --config configs/sr/sr3.yaml
```

> SR3 默认 50 步扩散（DDPM 是 500 步），因为条件信息降低了任务难度。GPU 上 10 epoch 约 5-8 分钟，CPU 上 15-25 分钟。

`models/sr3.py` 整个类只有十几行——核心就是把 `in_channels` 从 1 改成 2（1 通道噪声图 + 1 通道条件图）。`SR3UNet` 继承 `UNetModel`，结构完全没变，只是输入通道数多了 1。

训练时条件图通过通道拼接注入：

```python
predicted_noise = self.model(torch.cat([x_noisy, lr], dim=1), t)
```

调度器直接复用 DDPM 的 `GaussianDiffusion`，没有重写。

SR3 的推理仍然是从纯噪声开始，逐步去噪，每一步都参考低清条件图。这和后面 ResShift 从低清图附近开始不同。

值得对比的事：

- SRResNet 一步出结果，SR3 多步出结果 — 哪个更清晰？
- SR3 的 `in_channels=2` 如果换成 RGB 图像就变成 `3+3=6`
- 读 `models/sr3.py` 全文，只有十几行，看看"条件扩散"的工程实现有多简洁

---

## 第六阶段：ResShift 少步扩散超分

SR3 证明了条件扩散能做超分。ResShift 追问的是另一个问题：恢复任务里输入已经带了大量信息，有必要从纯噪声慢慢采样吗？

```bash
python main.py --config configs/sr/resshift.yaml
```

> ResShift 只需 15 步采样，比 SR3（50 步）和 DDPM（500 步）快很多。GPU 上 10 epoch 约 3-5 分钟，CPU 上 10-15 分钟。

ResShift 的核心变化：

- **预测目标变了**：不再预测噪声，而是预测残差 `R = HR - LR_up`
- **采样起点变了**：不再从纯噪声开始，而是从低清图附近开始
- **调度器换了**：用 `ResidualShiftScheduler` 替代 `GaussianDiffusion`

残差缩放调度的直觉：`t=0` 时残差保留完整（中间态接近目标图），`t=T` 时残差几乎消失（中间态接近低清图加噪声）。模型要学的是在任意时间步预测残差。

推理时从 `condition + noise` 开始，逐步恢复残差，最后一步直接用预测残差重建，不加噪声。

可以试的事：

- 把 `timesteps` 从 15 改成 5 或 30 — 观察步数对效果的影响
- 把 `noise_level` 从 0.15 改成 0.05 或 0.3 — 噪声强度的影响
- 把 `schedule` 从 `cosine` 改成 `linear` — 残差衰减策略的差异
- 同时跑 SRResNet、SR3、ResShift，对比 PSNR 和训练时间

三种超分方法的区别：

- **SRResNet**：一步直接修图，快但表达能力有限
- **SR3**：标准条件扩散，从纯噪声开始，步数较多
- **ResShift**：围绕残差设计的少步恢复，从低清图附近开始

---

## 第七阶段：工程实践与扩展

前六步是从模型角度学的。这一步是从工程角度看整个项目。

几个值得练手的事：

**换个数据集**：把 `cnn.yaml` 的 `data.dataset` 改成 `fashion_mnist`，不改任何代码直接跑。这就是配置驱动的好处。

**对比不同 latent_dim**：VAE 里分别试 `latent_dim: 2, 10, 20, 100`，用 `inference_vae.py` 生成图像看差异。

**给 DDPM 换噪声调度**：对比 `linear` 和 `cosine`，修改 timesteps 为 100 / 500 / 1000。

**写一份三方法对比报告**：SRResNet、SR3、ResShift 在相同数据集上训练，记录 PSNR、训练时间、推理时间。

**扩展 BaseTrainer**：增加学习率调度器支持——在 `_build_optimizer()` 后新增 `_build_scheduler()`，在 `fit()` 的每个 epoch 后调用 `scheduler.step()`。

**增加 SSIM 指标**：在 `utils/metrics.py` 中新增 `calculate_ssim()`，在 `BaseSRTrainer.evaluate()` 中同时计算 PSNR 和 SSIM。

**新增一个 trainer**：比如去噪任务——继承 `BaseTrainer`，覆写 `_build_criterion()`、`train_one_epoch()`、`evaluate()`，在 `TRAINER_REGISTRY` 中注册，写一个新的 yaml 配置。

真正学会工程，不是把现有代码跑通，而是你能在现有结构上继续扩展。

---

## 速查表

| 阶段 | 输入 | 目标 | 指标 | 采样方式 |
|------|------|------|------|----------|
| CNN | `(image)` | 交叉熵 | 准确率 ↑ | 无 |
| VAE | `(image)` | 重构 + KL | Loss ↓ | 采样 z → 解码 |
| DDPM | `(x_t, t)` | 噪声预测 | Noise Loss ↓ | T 步去噪 |
| SRResNet | `(lr_up)` | L1 残差 | PSNR ↑ | 单步前向 |
| SR3 | `([x_t, lr], t)` | 噪声预测 | PSNR ↑ | T 步条件去噪 |
| ResShift | `([shifted, lr], t)` | 残差预测 | PSNR ↑ | 少步残差恢复 |
