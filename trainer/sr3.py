import torch
import torch.nn as nn

from models.ddpm.diffusion import GaussianDiffusion
from trainer.sr import BaseSRTrainer
from utils.logger import log


class SR3Trainer(BaseSRTrainer):
    """教学版 SR3 训练器。

    它是 DDPM 和 ResShift 之间最重要的桥梁：
    - 和 DDPM 一样，训练目标是预测噪声
    - 和超分任务一样，模型带有低清条件图
    - 和 ResShift 相比，它仍然是更“标准”的条件扩散形式
    """

    def __init__(self, config, train_loader, val_loader=None):
        diff_cfg = config.get("diffusion", {})
        self.diffusion = GaussianDiffusion(
            timesteps=diff_cfg.get("timesteps", 100),
            beta_schedule=diff_cfg.get("schedule", "linear"),
        )
        super().__init__(config, train_loader, val_loader)

    def _build_criterion(self):
        return nn.MSELoss()

    def _diffusion_clip_range(self):
        return self._clamp_range()

    def _predict_noise(self, x_noisy, lr, t):
        return self.model(torch.cat([x_noisy, lr], dim=1), t)

    def _noise_prediction_loss(self, lr, hr):
        batch_size = hr.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(hr)
        x_noisy = self.diffusion.q_sample(x_start=hr, t=t, noise=noise)
        predicted_noise = self._predict_noise(x_noisy, lr, t)
        return self.criterion(predicted_noise, noise)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        log_interval = max(self.cfg.get("train", {}).get("log_interval", 50), 1)

        for batch_idx, (lr, hr) in enumerate(self.train_loader, start=1):
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            loss = self._noise_prediction_loss(lr, hr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = lr.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(self.train_loader):
                log(
                    f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"SR3 Noise MSE: {loss.item():.6f}"
                )

        avg_loss = total_loss / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Train SR3 Noise Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def infer(self, lr):
        current = torch.randn_like(lr)
        batch_size = lr.shape[0]

        for step in reversed(range(self.diffusion.timesteps)):
            t = torch.full((batch_size,), step, device=self.device, dtype=torch.long)
            predicted_noise = self._predict_noise(current, lr, t)
            current = self.diffusion.p_sample_from_pred_noise(
                current,
                t,
                predicted_noise,
                clip_denoised_range=self._diffusion_clip_range(),
            )

        return current.clamp(*self._clamp_range())
