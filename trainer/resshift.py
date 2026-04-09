import torch
import torch.nn as nn

from models.resshift import ResidualShiftScheduler
from utils.logger import log
from trainer.sr import BaseSRTrainer


class ResShiftTrainer(BaseSRTrainer):
    def __init__(self, config, train_loader, val_loader=None):
        resshift_cfg = config.get("resshift", {})
        self.scheduler = ResidualShiftScheduler(
            timesteps=resshift_cfg.get("timesteps", 15),
            noise_level=resshift_cfg.get("noise_level", 0.2),
            schedule=resshift_cfg.get("schedule", "linear"),
        )
        super().__init__(config, train_loader, val_loader)

    def _build_criterion(self):
        return nn.MSELoss()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for lr, hr in self.train_loader:
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            batch_size = lr.size(0)
            t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device)
            shifted, residual = self.scheduler.q_sample(hr, lr, t)
            predicted_residual = self.model(torch.cat([shifted, lr], dim=1), t)
            loss = self.criterion(predicted_residual, residual)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Train Residual MSE: {avg_loss:.6f}")
        return avg_loss

    def infer(self, lr):
        return self.scheduler.sample(
            self.model,
            lr,
            clamp_range=self._clamp_range(),
        )
