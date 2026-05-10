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
    """ResShift scheduler following the paper's residual-shifting process.

    The UNet takes ([x_t, y_0], t) and predicts x_0 directly.
    """

    def __init__(
        self,
        timesteps=15,
        noise_level=0.2,
        schedule="geometric",
        shift_power=0.3,
        eta_start=None,
        eta_end=0.999,
    ):
        if timesteps < 2:
            raise ValueError("ResShift timesteps must be >= 2.")
        if noise_level <= 0:
            raise ValueError("ResShift noise_level must be > 0.")

        self.timesteps = timesteps
        self.kappa = float(noise_level)
        self.schedule = schedule
        self.shift_power = float(shift_power)
        self.eta_start = (
            float(eta_start)
            if eta_start is not None
            else min((0.04 / self.kappa) ** 2, 0.001)
        )
        self.eta_end = float(eta_end)

        if not (0.0 < self.eta_start < self.eta_end < 1.0):
            raise ValueError("ResShift eta values must satisfy 0 < eta_start < eta_end < 1.")

        etas = self._build_eta_schedule()
        self.etas = etas.float()
        self.posterior_variance = self._build_posterior_variance().float()

    def _build_eta_schedule(self):
        if self.schedule == "geometric":
            return self._build_geometric_eta_schedule()
        if self.schedule == "linear":
            return torch.linspace(self.eta_start, self.eta_end, self.timesteps)
        if self.schedule == "cosine":
            steps = torch.linspace(0.0, 1.0, self.timesteps)
            weights = 1.0 - torch.cos(steps * math.pi / 2.0)
            return self.eta_start + (self.eta_end - self.eta_start) * weights

        raise ValueError(
            f"Unsupported ResShift schedule '{self.schedule}'. "
            "Use 'geometric', 'linear', or 'cosine'."
        )

    def _build_geometric_eta_schedule(self):
        if self.timesteps == 2:
            return torch.tensor([self.eta_start, self.eta_end], dtype=torch.float32)

        eta = torch.empty(self.timesteps, dtype=torch.float32)
        eta[0] = self.eta_start
        eta[-1] = self.eta_end

        base = math.exp(math.log(self.eta_end / self.eta_start) / (2.0 * (self.timesteps - 1)))
        sqrt_eta_start = math.sqrt(self.eta_start)

        for index in range(1, self.timesteps - 1):
            timestep = index + 1  # paper index starts from 1
            beta_t = ((timestep - 1) / (self.timesteps - 1)) ** self.shift_power * (
                self.timesteps - 1
            )
            sqrt_eta_t = sqrt_eta_start * (base ** beta_t)
            eta[index] = sqrt_eta_t**2

        return eta

    def _build_posterior_variance(self):
        variance = torch.zeros(self.timesteps, dtype=torch.float32)
        variance[0] = 0.0
        eta_t = self.etas[1:]
        eta_prev = self.etas[:-1]
        alpha_t = eta_t - eta_prev
        variance[1:] = (self.kappa**2) * (eta_prev / eta_t) * alpha_t
        return variance

    def _extract(self, values, t, x_shape):
        out = values.to(t.device).gather(0, t).float()
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))

    def q_sample(self, target, condition, t, noise=None):
        if noise is None:
            noise = torch.randn_like(target)

        eta_t = self._extract(self.etas, t, target.shape)
        shifted = target + eta_t * (condition - target) + self.kappa * torch.sqrt(eta_t) * noise
        return shifted, target

    def p_mean_variance(self, current, predicted_x0, t):
        if torch.any(t <= 0):
            raise ValueError("p_mean_variance expects timesteps > 0.")

        eta_t = self._extract(self.etas, t, current.shape)
        eta_prev = self._extract(self.etas, t - 1, current.shape)
        alpha_t = eta_t - eta_prev
        mean = (eta_prev / eta_t) * current + (alpha_t / eta_t) * predicted_x0
        variance = self._extract(self.posterior_variance, t, current.shape)
        return mean, variance

    @torch.no_grad()
    def sample(self, model, condition, clamp_range=None):
        batch_size = condition.shape[0]
        device = condition.device
        current = condition + self.kappa * torch.randn_like(condition)

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
