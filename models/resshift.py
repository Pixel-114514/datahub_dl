import math

import torch

from models.ddpm.unet import UNetModel


class ResShiftUNet(UNetModel):
    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )


class ResidualShiftScheduler:
    def __init__(self, timesteps=15, noise_level=0.2, schedule="linear"):
        if timesteps < 2:
            raise ValueError("ResShift timesteps must be >= 2.")

        self.timesteps = timesteps
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
        out = values.to(t.device).gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, target, condition, t, noise=None):
        residual = target - condition
        if noise is None:
            noise = torch.randn_like(target)

        residual_scale = self._extract(self.residual_scales, t, target.shape)
        noise_scale = self._extract(self.noise_scales, t, target.shape)
        shifted = condition + residual_scale * residual + noise_scale * noise
        return shifted, residual

    @torch.no_grad()
    def sample(self, model, condition, clamp_range=None):
        batch_size = condition.shape[0]
        device = condition.device
        noise_scale = self.noise_scales[-1].to(device)
        current = condition + noise_scale * torch.randn_like(condition)

        for step in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            predicted_residual = model(torch.cat([current, condition], dim=1), t)

            if step == 0:
                current = condition + predicted_residual
                continue

            prev_t = torch.full((batch_size,), step - 1, device=device, dtype=torch.long)
            residual_scale = self._extract(self.residual_scales, prev_t, current.shape)
            noise_scale = self._extract(self.noise_scales, prev_t, current.shape)
            current = condition + residual_scale * predicted_residual
            current = current + noise_scale * torch.randn_like(current)

        if clamp_range is not None:
            current = current.clamp(*clamp_range)
        return current
