import torch
import torch.nn as nn

from models.resshift import ResidualShiftScheduler
from trainer.sr import BaseSRTrainer
from utils.logger import log


class ResShiftTrainer(BaseSRTrainer):

    def __init__(self, config, train_loader, val_loader=None):
        #读取参数
        rs_cfg = config.get("resshift", {})
        self.scheduler = ResidualShiftScheduler(
            timesteps=rs_cfg.get("timesteps", 15),
            noise_level=rs_cfg.get("noise_level", 0.2),
            schedule=rs_cfg.get("schedule", "cosine"),
        )
        super().__init__(config, train_loader, val_loader)

    def _build_criterion(self):
        return nn.MSELoss()

    def _residual_loss(self, lr, hr):
        batch_size = hr.shape[0]
        # 随机采样时间步
        t = torch.randint(0, self.scheduler.timesteps, (batch_size,), device=self.device).long()

        # 前向过程：生成时间步 t的中间态，以及真实残差
        shifted, residual = self.scheduler.q_sample(hr, lr, t)

        # 把中间态和 LR 条件拼在一起送入网络，让网络预测残差
        predicted_residual = self.model(torch.cat([shifted, lr], dim=1), t)

        # 损失设预测残差和真实残差之间的 MSE
        return self.criterion(predicted_residual, residual)

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        log_interval = max(self.cfg.get("train", {}).get("log_interval", 50), 1)

        for batch_idx, (lr, hr) in enumerate(self.train_loader, start=1):
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            loss = self._residual_loss(lr, hr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = lr.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(self.train_loader):
                log(
                    f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"ResShift Residual MSE: {loss.item():.6f}"
                )

        avg_loss = total_loss / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Train ResShift Residual Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def infer(self, lr):
        # 逆向采样：从 LR + 少量噪声出发，经过 timesteps 步逐步还原 HR
        return self.scheduler.sample(
            self.model,
            lr,
            clamp_range=self._clamp_range(),
        )
