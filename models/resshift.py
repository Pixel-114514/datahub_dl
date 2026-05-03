import math

import torch

from models.ddpm.unet import UNetModel


# in_channels=2：把当前中间态和LR通道维拼在一起
# 网络输出预测残差（超分和低分间的），而不是噪声
class ResShiftUNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )


#从纯噪声出发，让图像从 LR 逐步向 HR 偏移只需 15步收敛
class ResidualShiftScheduler:
    def __init__(self, timesteps=15, noise_level=0.2, schedule="linear"):
        if timesteps < 2:
            raise ValueError("ResShift timesteps must be >= 2.")

        self.timesteps = timesteps

        # residual_scales每步残差比例
        #noise_scales每步噪声强度
        if schedule == "linear":
            residual_scales = torch.linspace(1.0, 0.0, timesteps)
        elif schedule == "cosine":
            residual_scales = torch.cos(torch.linspace(0.0, math.pi / 2.0, timesteps))
        else:
            raise ValueError(
                f"Unsupported ResShift schedule '{schedule}'. "
                "Use 'linear' or 'cosine'."
            )

        self.residual_scales = residual_scales.float()
        self.noise_scales = (1.0 - residual_scales).float() * noise_level

    def _extract(self, values, t, x_shape):
        # 按时间步t从调度表里取出对应标量
        out = values.to(t.device).gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, target, condition, t, noise=None):
        # 前向，给定超分和低分，生成时间步 t的中间态
        #训练时同时返回残差，作为模型的学习目标
        residual = target - condition
        if noise is None:
            noise = torch.randn_like(target)

        residual_scale = self._extract(self.residual_scales, t, target.shape)
        noise_scale = self._extract(self.noise_scales, t, target.shape)
        shifted = condition + residual_scale * residual + noise_scale * noise
        return shifted, residual

    @torch.no_grad()
    def sample(self, model, condition, clamp_range=None):
        # 逆向，从LR +少量噪声出发，逐步去噪，最终恢复超分
        batch_size = condition.shape[0]
        device = condition.device

        #起点是·LR 加上最后一步对应的噪声
        noise_scale = self.noise_scales[-1].to(device)
        current = condition + noise_scale * torch.randn_like(condition)

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)

            # 网络预测当前步的残差
            predicted_residual = model(torch.cat([current, condition], dim=1), t)

            if step == 0:
                #直接用预测残差还原 HR，不再加噪声
                current = condition + predicted_residual
                continue

            # 中间步，用上一步的调度系数更新，用少量噪声保持随机性
            prev_t = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
            residual_scale = self._extract(self.residual_scales, prev_t, current.shape)
            noise_scale = self._extract(self.noise_scales, prev_t, current.shape)
            current = condition + residual_scale * predicted_residual
            current = current + noise_scale * torch.randn_like(current)

        if clamp_range is not None:
            current = current.clamp(*clamp_range)
        return current
