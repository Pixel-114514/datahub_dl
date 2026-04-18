# 学习路径

这份文档不是在讲某一个模型，而是在讲整条学习路径该怎么组织。

目标不是一次性理解所有论文细节，而是沿着一条连续的路径，从最基础的深度学习训练，一步一步走到扩散超分和工程实践。

推荐主线如下：

1. 手写数字识别
2. VAE 图片生成
3. DDPM 基础扩散
4. 超分基线
5. SR3 条件扩散超分
6. ResShift 少步扩散超分
7. 工程实践与扩展练习

---

## 第一阶段：手写数字识别

目标不是"刷高分"，而是先完整跑通一个深度学习项目。

### 这一阶段要掌握什么

| 概念 | 含义 | 在代码中的体现 |
|------|------|----------------|
| 样本 (sample) | 一张图片就是一个样本 | `data/dataloader.py` 中 `get_dataloader()` 返回的每个 batch 元素 |
| 标签 (label) | 图片对应的类别编号 | MNIST 中 0-9 的整数，和图片一起组成 `(image, label)` 元组 |
| batch | 一次送入模型的多个样本 | 配置文件中 `train.batch_size: 128`，即每次取 128 张图 |
| 前向传播 | 数据从输入层经过各层计算到输出 | `models/cnn.py` 中 `forward()` 方法，输入 `(B,1,28,28)` 输出 `(B,10)` |
| loss | 模型预测与真实标签的差距 | `trainer/base.py` 中 `nn.CrossEntropyLoss()`，值越小越好 |
| 反向传播 | 根据 loss 计算各参数的梯度 | `loss.backward()` 这一行，PyTorch 自动完成 |
| 优化器 | 根据梯度更新模型参数 | `trainer/base.py` 中 `optim.Adam()`，`optimizer.step()` 执行更新 |
| 训练集 / 验证集 | 训练集用于学习，验证集用于检测泛化能力 | `get_dataloader()` 返回两个 loader，防止模型"死记硬背" |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/classification/cnn.yaml` | 实验配置 | 每个字段控制什么，修改后效果如何变化 |
| `main.py` | 总入口 | 配置如何被读取、trainer 如何被选择、训练如何被启动 |
| `trainer/base.py` | 基础训练器 | `train_one_epoch()` 和 `evaluate()` 的完整流程 |
| `models/cnn.py` | CNN 模型 | `forward()` 的输入输出形状、卷积层如何堆叠 |

### 运行命令与预期输出

```bash
# 基础运行
python main.py --config configs/classification/cnn.yaml

# 如果没有 GPU，指定 CPU
python main.py --config configs/classification/cnn.yaml --device cpu
```

预期输出类似：

```
PyTorch 版本: 2.x.x
CUDA 是否可用: True
...
Loading config from: configs/classification/cnn.yaml
Using seed: 42
Using device: cuda
Initializing trainer: base (BaseTrainer)
Epoch [1] Train Loss: 0.4521
Epoch [1] Val Acc: 0.9210
...
Epoch [10] Train Loss: 0.0132
Epoch [10] Val Acc: 0.9912
Training finished. Best Val Acc: 0.9921 @ epoch 8
Experiment directory: checkpoints/cnn
```

> **提示**：首次运行会自动下载 MNIST 数据集到 `./data` 目录，约 11MB。

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码位置 |
|----------|------|----------|
| 配置驱动训练 | 所有超参数都在 yaml 中定义，不用改代码就能换实验 | `configs/classification/cnn.yaml` |
| 入口文件保持薄 | `main.py` 只做"读配置 → 选 trainer → 启动训练"三件事 | `main.py` 总共不到 60 行有效代码 |
| checkpoint 保存 | 每个 epoch 保存 `last.pth`，最优结果保存 `best.pth` | `trainer/base.py` 的 `save_checkpoint()` |
| 实验目录管理 | 每次实验自动创建独立目录，配置也会被保存为 `config.json` | `trainer/base.py` 的 `__init__()` |
| trainer 和 model 分离 | trainer 管"怎么训"，model 管"网络长什么样" | `trainer/base.py` vs `models/cnn.py` |

### 可以直接做的动作

1. **先跑 `cnn.yaml`**：观察训练 loss 下降和验证准确率上升的过程
2. **再跑 `resnet.yaml`**：`python main.py --config configs/classification/resnet.yaml`
3. **比较两个模型**：
   - CNN 参数量约 7 万，ResNet 参数量更大
   - 在 `models/cnn.py` 和 `models/resnet.py` 中对比网络结构差异
   - 注意 ResNet 引入了残差连接（`x + self.block(x)`）
4. **修改超参数观察变化**：
   - 把 `batch_size` 从 128 改成 32：训练更慢但梯度估计更准
   - 把 `lr` 从 0.001 改成 0.01：可能训练更快，但也可能震荡甚至不收敛
   - 把 `epochs` 从 10 改成 3：观察欠拟合的表现

### 常见问题

**Q: 为什么验证准确率比训练准确率低？**
A: 这是正常现象。模型在训练集上"见过"这些数据，但在验证集上没见过，性能自然会有差距。如果差距很大（比如训练 99% 验证 85%），说明过拟合了。

**Q: 为什么每次跑结果不完全一样？**
A: 配置中 `seed: 42` 控制随机种子，同 seed 同设备结果应一致。但如果换了设备（CPU vs GPU）或 PyTorch 版本，浮点运算差异会导致结果略有不同。

---

## 第二阶段：VAE 图片生成

这一阶段的作用是把视角从"判别任务"带到"生成任务"。

### 这里要完成的认知转变

分类任务关心：

- 这张图是什么类别 → 输出一个类别编号

VAE 关心：

- 如何把图像压缩到潜空间 → 编码器输出 `mu` 和 `logvar`
- 如何从潜空间重建出图像 → 解码器从 `z` 恢复图像
- 如何从潜变量中采样生成新图 → 随机采样 `z`，解码器直接生成

### 这一阶段要掌握什么

| 概念 | 直觉理解 | 在代码中的体现 |
|------|----------|----------------|
| `mu` | 潜变量分布的均值，"这张图大概在潜空间的哪个位置" | `models/vae.py` 中 `self.fc_mu(h)` |
| `logvar` | 潜变量分布的方差的对数，"这个位置的不确定性有多大" | `models/vae.py` 中 `self.fc_logvar(h)` |
| 重参数化 | 把不可导的采样操作改写成可导形式，让梯度能传回去 | `models/vae.py` 中 `z = mu + eps * std` |
| 重构损失 | 解码出来的图和原图有多像 | `trainer/vae.py` 中 `binary_cross_entropy(x_recon, x)` |
| KL 损失 | 潜变量分布离标准正态有多远 | `trainer/vae.py` 中 `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))` |
| 潜空间 | 图像被压缩后的低维表示空间 | 配置中 `latent_dim: 20`，即 20 维向量 |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/generate/vae.yaml` | VAE 实验配置 | `latent_dim: 20` 控制潜空间维度 |
| `models/vae.py` | VAE 模型 | `encode()` → `reparameterize()` → `decode()` 的完整链路 |
| `trainer/vae.py` | VAE 训练器 | loss 为什么返回三项（总 loss / 重构 / KL） |
| `inference_vae.py` | 推理脚本 | 如何从潜空间随机采样并生成图像 |

### 运行命令与预期输出

```bash
# 训练 VAE
python main.py --config configs/generate/vae.yaml

# 训练完成后，用推理脚本生成图像
python inference_vae.py
```

预期训练输出：

```
Epoch [1] Train Loss: 258.1234 | Recon: 254.5678 | KL: 3.5556
Epoch [1] Val Loss: 245.8901
...
Epoch [70] Train Loss: 142.3456 | Recon: 135.7890 | KL: 6.5566
```

> **注意**：VAE 的 loss 是"越小越好"，和分类任务的准确率"越大越好"相反。这在 `trainer/vae.py` 中通过 `_monitor_mode: "min"` 来控制。

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码对比 |
|----------|------|----------|
| trainer 覆写基类方法 | VAE 不用基类的 `CrossEntropyLoss`，而是覆写 `_build_criterion()` | `trainer/vae.py` vs `trainer/base.py` |
| 模型输出不只是一个张量 | VAE 的 `forward()` 返回 `(x_recon, mu, logvar)` 三元组 | `models/vae.py` 最后一行 |
| 训练和推理逻辑分离 | 训练用 `main.py`，推理用 `inference_vae.py` | 两个独立脚本 |
| 监控指标方向不同 | 分类看 `val_acc`（越大越好），VAE 看 `val_loss`（越小越好） | `_monitor_mode()` 返回值不同 |

### 可以直接做的动作

1. **跑通 VAE 训练**：`python main.py --config configs/generate/vae.yaml`
2. **生成图像**：训练完成后运行 `python inference_vae.py`
3. **修改 `latent_dim`**：从 20 改成 2，观察生成质量变化（维度太低信息不够，太高则潜空间不够紧凑）
4. **观察 KL 项的变化**：训练初期 KL 很小，随着训练推进 KL 会逐渐增大，说明潜空间在被"整理"

### 常见问题

**Q: 为什么 VAE 生成的图像比较模糊？**
A: 这是 VAE 的固有局限。因为 VAE 本质上是在优化"平均"的输出——它学到的是数据分布的均值，而均值天然是模糊的。扩散模型（后续阶段）在生成质量上通常优于 VAE。

**Q: 重构损失为什么用 `binary_cross_entropy` 而不是 `MSE`？**
A: 当图像像素值在 [0, 1] 范围时，BCE 等价于伯努利对数似然，理论更自然。MSE 也可以用，但效果差异不大。本项目选择 BCE 是因为更常见。

---

## 第三阶段：DDPM 基础扩散

这一阶段是把 VAE 的"潜空间建模"进一步推进到"过程建模"。

### 这里要完成的认知转变

VAE 更像：

- 一次编码：图像 → 潜变量 `z`
- 一次解码：`z` → 重建图像
- 过程是"一步到位"的

DDPM 更像：

- 正向：一张图逐步加噪，经过 T 步后变成纯噪声
- 反向：从纯噪声开始，逐步去噪，经过 T 步后恢复出图像
- 过程是"多步迭代"的

### 这一阶段要掌握什么

| 概念 | 直觉理解 | 在代码中的体现 |
|------|----------|----------------|
| 时间步 `t` | 表示当前图像被加噪到什么程度，`t=0` 是原图，`t=T` 是纯噪声 | `models/ddpm/diffusion.py` 中 `q_sample(x_start, t)` |
| 正向扩散 `q_sample` | 给原图加噪得到 `x_t`，公式：`x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε` | `GaussianDiffusion.q_sample()` |
| 反向去噪 `p_sample` | 给定 `x_t` 和预测噪声，推算 `x_{t-1}` | `GaussianDiffusion.p_sample()` |
| 噪声预测 | 模型的训练目标不是预测图像，而是预测"加了什么噪声" | `DiffusionTrainer._compute_noise_prediction_loss()` |
| beta schedule | 控制每一步加噪的强度，有 `linear` 和 `cosine` 两种 | `GaussianDiffusion.__init__()` 中的 `beta_schedule` 参数 |
| UNet | 扩散模型常用的网络结构，有下采样-瓶颈-上采样的对称结构 | `models/ddpm/unet.py` 中的 `UNetModel` |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/generate/ddpm.yaml` | DDPM 实验配置 | `diffusion.timesteps: 500` 和 `diffusion.schedule: linear` |
| `models/ddpm/unet.py` | UNet 网络 | `timestep_embedding()` 如何把时间步编码成向量；`forward(x, timesteps)` 的输入输出 |
| `models/ddpm/diffusion.py` | 扩散调度器 | `q_sample()` 加噪、`p_sample()` 去噪、`_extract()` 取出对应时间步的系数 |
| `trainer/diffusion.py` | DDPM 训练器 | 训练时随机采样 `t`，验证时额外保存采样图片 |

### 运行命令与预期输出

```bash
python main.py --config configs/generate/ddpm.yaml
```

> **重要提示**：DDPM 训练收敛较慢，默认配置只设了 1 个 epoch。建议把 `epochs` 改到 30-50 才能看到较好的生成效果。首次运行采样过程会比较慢（500 步去噪），请耐心等待。

预期输出：

```
Epoch [1] Batch [1/xxx] Noise MSE: 0.823456
Epoch [1] Batch [50/xxx] Noise MSE: 0.312789
...
Epoch [1] Train Noise Loss: 0.285431
Epoch [1] Val Noise Loss: 0.271234
Saved diffusion samples to checkpoints/ddpm_mnist_v1/sample_epoch_1.png
```

> **提示**：每个 epoch 结束后，训练器会自动保存一张采样图片到实验目录，你可以直观看到模型当前能生成什么样的图像。

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码位置 |
|----------|------|----------|
| 网络和调度器分离 | UNet 只管预测噪声，GaussianDiffusion 只管加噪/去噪公式 | `models/ddpm/unet.py` vs `models/ddpm/diffusion.py` |
| 训练目标不是最终输出 | 模型预测的是噪声，不是图像本身 | `DiffusionTrainer._compute_noise_prediction_loss()` |
| 指标和可视化并行 | 验证时既计算 loss 指标，又保存采样图片 | `DiffusionTrainer.evaluate()` 中两件事分开做 |
| 配置驱动的扩散参数 | `timesteps` 和 `schedule` 都在 yaml 中配置 | `configs/generate/ddpm.yaml` 的 `diffusion` 段 |

### 可以直接做的动作

1. **跑通 DDPM 训练**：先跑 1 个 epoch 确认流程没问题，再把 epochs 调大
2. **观察采样图片**：打开 `checkpoints/ddpm_mnist_v1/sample_epoch_*.png`，早期是噪声，后期逐渐成形
3. **修改 timesteps**：从 500 改成 100，训练更快但生成质量可能下降
4. **修改 schedule**：从 `linear` 改成 `cosine`，观察训练曲线和生成效果的差异
5. **修改 image_size**：从 28 改成 32，注意 UNet 的 `attention_resolutions` 也需要相应调整

### 常见问题

**Q: 为什么 DDPM 训练时不是从纯噪声开始走完整个采样链？**
A: 训练时是随机抽一个时间步 `t`，只训练模型在这个 `t` 上的去噪能力。这比走完整个链路高效得多，而且等价于让模型学会所有时间步的去噪。完整采样只在推理时才做。

**Q: 为什么采样这么慢？**
A: 因为推理时要从 `t=T` 一步步去噪到 `t=0`，每一步都要跑一次模型前向传播。T=500 就要跑 500 次。这就是为什么后续 ResShift 要设计"少步采样"。

**Q: `value_range: minus_one_one` 是什么意思？**
A: 扩散模型通常把图像像素从 [0,1] 线性映射到 [-1,1]，这样加噪后的值域更对称，训练更稳定。在 `data/transforms.py` 中通过 `x * 2.0 - 1.0` 实现。

---

## 第四阶段：超分基线

这一阶段的作用是从"无条件生成"转向"图像恢复任务"。

### 这里要完成的认知转变

以前是：

- 分类：从图像到类别 → `(image, label)`
- 生成：从噪声到图像 → 学习数据分布

现在变成：

- 超分：已经有低质量图，目标是恢复出高质量图 → `(lr_up, hr)`

### 这一阶段要掌握什么

| 概念 | 直觉理解 | 在代码中的体现 |
|------|----------|----------------|
| `(lr_up, hr)` 训练对 | `lr_up` 是低清图插值回原尺寸，`hr` 是原始高清图 | `data/sr_dataset.py` 中 `__getitem__()` 返回值 |
| 下采样-上采样流程 | `hr → 下采样得到 lr → 插值回原尺寸得到 lr_up` | `SyntheticSuperResolutionDataset.__getitem()` |
| PSNR | 峰值信噪比，衡量恢复图和真值图的像素级相似度，越高越好 | `utils/metrics.py` 中 `calculate_psnr()` |
| 残差学习 | 网络不直接输出超分图，而是输出 `残差 = 预测 - 输入`，最终结果 = 输入 + 残差 | `models/sr.py` 中 `return x + residual` |
| L1 Loss | 绝对值误差，超分任务常用，比 MSE 更不容易模糊 | `trainer/sr.py` 中 `nn.L1Loss()` |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/sr/srresnet.yaml` | 超分实验配置 | `data.task: super_resolution` 触发数据包装 |
| `data/sr_dataset.py` | 超分数据包装 | `__getitem__()` 如何把 `(image, label)` 变成 `(lr_up, hr)` |
| `models/sr.py` | SRResNet 模型 | 残差块结构、全局跳跃连接 `x + residual` |
| `trainer/sr.py` | 超分训练器 | `BaseSRTrainer` 是 SR3 和 ResShift 的父类 |

### 运行命令与预期输出

```bash
python main.py --config configs/sr/srresnet.yaml
```

预期输出：

```
Epoch [1] Train L1: 0.082345
Epoch [1] Val PSNR: 18.5432 dB
...
Epoch [10] Train L1: 0.031267
Epoch [10] Val PSNR: 22.8765 dB
Saved super-resolution samples to checkpoints/sr_baseline_mnist/sr_epoch_10.png
```

> **提示**：超分可视化图片会同时展示低清输入、超分结果、高清真值三行，方便直观对比。

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码位置 |
|----------|------|----------|
| 原始数据集和任务样本分离 | 同一个 MNIST 可以被包装成分类样本或超分样本 | `data/dataloader.py` 中 `_infer_task()` 判断任务类型 |
| 指标随任务改变 | 分类看准确率，超分看 PSNR | `BaseSRTrainer._monitor_name()` 返回 `val_psnr` |
| 可视化保存 | 每个 epoch 保存低清/超分/高清对比图 | `BaseSRTrainer._save_visuals()` |
| 训练器继承体系 | `BaseSRTrainer` 继承 `BaseTrainer`，`SuperResolutionTrainer` 继承 `BaseSRTrainer` | `trainer/sr.py` 的类定义 |

### 可以直接做的动作

1. **跑通超分训练**：`python main.py --config configs/sr/srresnet.yaml`
2. **观察可视化结果**：打开 `checkpoints/sr_baseline_mnist/sr_epoch_*.png`，对比三行图像
3. **修改 scale_factor**：从 2 改成 4，观察超分难度增加后 PSNR 的变化
4. **修改 num_blocks**：从 6 改成 3，观察更浅网络的表现差异
5. **添加噪声退化**：设置 `noise_std: 0.01`，让低清图带噪声，观察恢复效果

### 常见问题

**Q: 为什么要学残差而不是直接学输出？**
A: 低清图已经包含了大部分低频信息（整体结构、颜色），真正缺失的是高频细节（边缘、纹理）。学残差让网络聚焦于"补细节"，而不是"从头生成整张图"，训练更容易收敛。

**Q: PSNR 多少算好？**
A: 这取决于任务难度和数据集。MNIST 上 2x 超分，SRResNet 通常能达到 22-25 dB。真实自然图像上 4x 超分，25-28 dB 就算不错了。PSNR 不是完美指标，但作为起点足够用。

---

## 第五阶段：SR3 条件扩散超分

这是整条路径里最重要的桥梁阶段。

如果从 DDPM 直接跳到 ResShift，跨度会很大。

SR3 的作用，就是先接受下面这件事：

- 扩散模型不只能做无条件生成
- 它也可以带条件输入做图像恢复

### 这里要完成的认知转变

DDPM：

- 输入：`(x_t, t)` → 预测噪声
- 从纯噪声逐步生成图像

SR3：

- 输入：`([x_t, lr], t)` → 预测噪声
- 给定低清条件图，逐步恢复出高清图

**关键变化只有一个**：模型输入从 `x_t` 变成了 `[x_t, lr]`，即把条件图和当前噪声图在通道维度上拼接。

### 这一阶段要掌握什么

| 概念 | 直觉理解 | 在代码中的体现 |
|------|----------|----------------|
| 条件扩散 | 扩散过程中引入额外条件信息 | SR3 把低清图拼接到噪声图上 |
| 通道拼接注入 | 把条件图和噪声图在通道维度 concat 后送入网络 | `models/sr3.py` 中 `in_channels=2`（1 通道噪声图 + 1 通道条件图） |
| 条件 UNet | 和普通 UNet 结构完全一样，只是输入通道数变了 | `SR3UNet` 继承 `UNetModel`，只改了 `in_channels` |
| SR3 的桥梁作用 | 它是 DDPM（无条件扩散）到 ResShift（少步残差恢复）的过渡 | 先理解"条件怎么注入"，再理解"过程怎么简化" |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/sr/sr3.yaml` | SR3 实验配置 | `model.params.in_channels: 2`（条件图占 1 个通道） |
| `models/sr3.py` | SR3 条件 UNet | 整个类只有几行，核心是 `in_channels=2` |
| `trainer/sr3.py` | SR3 训练器 | `_predict_noise()` 中 `torch.cat([x_noisy, lr], dim=1)` 是条件注入的关键 |
| `models/ddpm/diffusion.py` | 调度器复用 | SR3 直接复用 DDPM 的 `GaussianDiffusion`，没有重写 |

### 运行命令与预期输出

```bash
python main.py --config configs/sr/sr3.yaml
```

> **提示**：SR3 默认只设了 50 步扩散（而非 DDPM 的 500 步），因为条件信息已经大幅降低了任务难度。但训练仍然需要较多 epoch 才能收敛。

预期输出：

```
Epoch [1] Batch [1/xxx] SR3 Noise MSE: 0.723456
...
Epoch [1] Train SR3 Noise Loss: 0.456789
Epoch [1] Val PSNR: 16.2345 dB
Saved super-resolution samples to checkpoints/sr3_mnist_toy/sr_epoch_1.png
```

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码位置 |
|----------|------|----------|
| 复用已有组件 | SR3 直接复用 `GaussianDiffusion`，不需要重写加噪/去噪逻辑 | `trainer/sr3.py` 中 `from models.ddpm.diffusion import GaussianDiffusion` |
| 新增 trainer 扩展任务 | 只需写一个新的 trainer 类，配置中指定 `trainer_name: sr3` | `trainer/sr3.py` |
| 条件信息通过拼接注入 | 最简单也最直观的条件注入方式 | `torch.cat([x_noisy, lr], dim=1)` |
| 继承体系 | `SR3Trainer` 继承 `BaseSRTrainer`，复用 PSNR 计算和可视化逻辑 | `trainer/sr3.py` |

### 可以直接做的动作

1. **跑通 SR3 训练**：`python main.py --config configs/sr/sr3.yaml`
2. **对比 SRResNet 和 SR3 的可视化**：哪个更清晰？哪个训练更慢？
3. **修改扩散步数**：从 50 改成 100 或 20，对比效果和速度
4. **阅读 `models/sr3.py`**：只有 16 行代码，理解"条件扩散"的工程实现有多简洁

### 常见问题

**Q: SR3 的 `in_channels=2` 是怎么来的？**
A: 1 个通道给当前噪声图 `x_t`，1 个通道给低清条件图 `lr`。两者在通道维度拼接后送入 UNet。如果输入是 RGB 图像，则 `in_channels = 3 + 3 = 6`。

**Q: 为什么 SR3 推理时从随机噪声开始，而不是从低清图开始？**
A: SR3 的设计是"以低清图为条件，从噪声恢复高清图"。低清图作为条件信息参与每一步去噪，但采样起点仍然是噪声。ResShift 会改变这一点——它从低清图附近开始，所以步数更少。

---

## 第六阶段：ResShift 少步扩散超分

这一阶段不是重新介绍一遍扩散，而是强调"为什么在恢复任务里要做更高效的改造"。

### 这里要完成的认知转变

SR3 已经说明：

- 条件扩散可以做超分

ResShift 进一步说明：

- 在恢复任务里没必要完全照搬无条件扩散
- 可以围绕残差和少步采样做更贴近任务的设计

### 这一阶段要掌握什么

| 概念 | 直觉理解 | 在代码中的体现 |
|------|----------|----------------|
| 残差 `R = HR - LR_up` | 高清图和低清图之间的差异，就是需要"补"的部分 | `models/resshift.py` 中 `residual = target - condition` |
| 残差缩放调度 | 不同时间步保留不同比例的残差，`t=0` 保留全部，`t=T` 残差为 0 | `ResidualShiftScheduler` 中 `residual_scales` 从 1.0 线性/余弦降到 0.0 |
| 噪声缩放 | 残差越少的地方加越多噪声，形成从"接近目标"到"接近噪声"的过渡 | `noise_scales = (1 - residual_scales) * noise_level` |
| 少步采样 | 只需 15 步而非 500 步，因为条件图已经提供了大量信息 | 配置中 `resshift.timesteps: 15` |
| 残差预测 | 模型预测的是残差 `R`，不是噪声 | `trainer/resshift.py` 中 `loss = criterion(predicted_residual, residual)` |

### 对应代码

| 文件 | 职责 | 重点看什么 |
|------|------|-----------|
| `configs/sr/resshift.yaml` | ResShift 实验配置 | `resshift.timesteps: 15`、`noise_level: 0.15` |
| `models/resshift.py` | 网络和调度器 | `ResidualShiftScheduler.q_sample()` 构造训练中间态；`sample()` 做逐步恢复 |
| `trainer/resshift.py` | ResShift 训练器 | 训练时 `scheduler.q_sample(hr, lr, t)` 构造中间态，推理时 `scheduler.sample(model, lr)` |

### 运行命令与预期输出

```bash
python main.py --config configs/sr/resshift.yaml
```

预期输出：

```
Epoch [1] Train Residual MSE: 0.123456
Epoch [1] Val PSNR: 19.5678 dB
...
```

> **提示**：ResShift 只需 15 步采样，推理速度比 SR3（50 步）和 DDPM（500 步）快很多。

### 这一阶段对应的工程实践

| 工程概念 | 说明 | 代码位置 |
|----------|------|----------|
| 在相同框架里迭代算法 | ResShift 复用 `BaseSRTrainer`，只换了调度器和训练逻辑 | `trainer/resshift.py` |
| 研究动机落到工程 | "恢复任务不需要从纯噪声开始" → 代码里从 `condition + noise` 开始 | `ResidualShiftScheduler.sample()` |
| 对比不同方案 | SRResNet（一步）/ SR3（多步条件扩散）/ ResShift（少步残差恢复） | 三个 trainer 共享 `BaseSRTrainer` 的评估逻辑 |

### 可以直接做的动作

1. **跑通 ResShift 训练**：`python main.py --config configs/sr/resshift.yaml`
2. **对比三种超分方法**：
   - SRResNet：`python main.py --config configs/sr/srresnet.yaml`
   - SR3：`python main.py --config configs/sr/sr3.yaml`
   - ResShift：`python main.py --config configs/sr/resshift.yaml`
   - 对比三者的 PSNR、可视化效果和训练时间
3. **修改 timesteps**：从 15 改成 5 或 30，观察步数对效果的影响
4. **修改 noise_level**：从 0.15 改成 0.05 或 0.3，观察噪声强度的影响
5. **修改 schedule**：从 `cosine` 改成 `linear`，观察残差衰减策略的差异

### 常见问题

**Q: ResShift 和 SR3 的根本区别是什么？**
A: SR3 预测的是噪声（和 DDPM 一样），从纯噪声开始采样；ResShift 预测的是残差，从低清图附近开始采样。这使得 ResShift 可以用更少的步数完成恢复。

**Q: 为什么 ResShift 只用 15 步就能工作？**
A: 因为低清条件图已经包含了图像的大部分信息，模型只需要补齐残差部分。而 DDPM/SR3 从纯噪声开始，需要更多步来"构建"整张图。

---

## 第七阶段：工程实践与扩展

前六个阶段学的是"从模型到任务"。

最后这一阶段要真正开始上手"做项目"。

### 应该强化的工程实践主题

| 主题 | 说明 | 对应代码 |
|------|------|----------|
| 配置驱动训练 | 所有超参数在 yaml 中定义，不改代码就能换实验 | `configs/` 目录下所有 yaml |
| trainer / model / data 分层 | 训练逻辑、网络结构、数据准备各司其职 | `trainer/`、`models/`、`data/` |
| 注册表模式 | 通过 `TRAINER_REGISTRY` 和 `MODEL_REGISTRY` 解耦入口和具体实现 | `trainer/__init__.py`、`models/__init__.py` |
| checkpoint 管理 | 自动保存 `best.pth` 和 `last.pth`，配置也被保存 | `trainer/base.py` 的 `save_checkpoint()` |
| 指标与可视化 | 不同任务有不同的监控指标和可视化方式 | `utils/metrics.py`、各 trainer 的 `evaluate()` |
| 随机种子与复现 | `seed: 42` 保证相同配置下结果可复现 | `utils/seed.py` |
| 任务扩展复用 | 新任务只需新增 trainer + model + config | SR3 和 ResShift 的添加过程就是范例 |

### 可以继续做的练习

1. **给分类任务增加 `FashionMNIST` 配置**：
   - 复制 `configs/classification/cnn.yaml`
   - 把 `data.dataset` 改成 `fashion_mnist`
   - 运行并对比 MNIST 和 FashionMNIST 上的准确率差异

2. **给 VAE 增加不同 latent_dim 的实验**：
   - 分别尝试 `latent_dim: 2, 10, 20, 100`
   - 用 `inference_vae.py` 生成图像，观察潜空间维度对生成质量的影响

3. **给 DDPM 改不同的噪声调度策略**：
   - 对比 `linear` 和 `cosine` 两种 schedule
   - 修改 timesteps 为 100 / 500 / 1000，对比生成效果和采样速度

4. **给 SR3 改不同的扩散步数**：
   - 尝试 `timesteps: 10, 50, 100, 200`
   - 对比效果和速度，找到效率与质量的平衡点

5. **对比 SRResNet、SR3、ResShift 的可视化结果和 PSNR**：
   - 三种方法在相同数据集上训练
   - 记录 PSNR、训练时间、推理时间
   - 写一份简单的对比报告

6. **尝试把超分数据从 MNIST 替换到 CIFAR10**：
   - 修改 yaml 中 `data.dataset: cifar10`
   - 注意 CIFAR10 是 3 通道 32x32，需要调整 `in_channels` 和 `image_size`

7. **给超分任务增加 SSIM 指标**：
   - 在 `utils/metrics.py` 中新增 `calculate_ssim()` 函数
   - 在 `BaseSRTrainer.evaluate()` 中同时计算 PSNR 和 SSIM

8. **给 BaseTrainer 增加学习率调度器支持**：
   - 在 `_build_optimizer()` 后新增 `_build_scheduler()`
   - 在 `fit()` 的每个 epoch 后调用 `scheduler.step()`

---

## 组织方式

如果按整条路径来排，比较自然的顺序是：

### 第一部分：深度学习入门

- CNN / ResNet 分类
- 标准训练流程（前向 → loss → 反向 → 更新）
- 配置和实验管理

### 第二部分：生成模型入门

- VAE
- 潜变量和生成
- 从判别到生成的认知转变

### 第三部分：扩散模型入门

- DDPM
- 时间步和噪声预测
- 多步采样过程

### 第四部分：图像恢复任务

- 超分基线（SRResNet）
- 数据包装和 PSNR
- 残差学习思想

### 第五部分：扩散超分

- SR3：条件扩散超分（桥梁阶段）
- ResShift：少步残差恢复超分

### 第六部分：工程实践

- 重构与扩展
- 对比实验
- 写自己的 trainer / model / config

---

## 一句总纲

如果要把整条路径压缩成一句话，可以直接概括成：

> 先用手写数字识别学会"怎么训练一个模型"，再用 VAE 学会"怎么生成图像"，再用 DDPM 学会"怎么逐步建模生成过程"，然后用 SR3 和 ResShift 学会"怎么把扩散思想用到条件超分和工程实现里"。

---

## 各阶段关键概念速查表

| 阶段 | 任务类型 | 模型输入 | 训练目标 | 评估指标 | 采样方式 |
|------|----------|----------|----------|----------|----------|
| CNN 分类 | 判别 | `(image)` | 交叉熵 | 准确率 ↑ | 无 |
| VAE 生成 | 生成 | `(image)` | 重构 + KL | Loss ↓ | 采样 z → 解码 |
| DDPM 扩散 | 生成 | `(x_t, t)` | 噪声预测 MSE | Noise Loss ↓ | T 步去噪 |
| SRResNet 超分 | 恢复 | `(lr_up)` | L1 残差 | PSNR ↑ | 单步前向 |
| SR3 条件扩散超分 | 条件恢复 | `([x_t, lr], t)` | 噪声预测 MSE | PSNR ↑ | T 步条件去噪 |
| ResShift 少步超分 | 条件恢复 | `([shifted, lr], t)` | 残差预测 MSE | PSNR ↑ | 少步残差恢复 |
