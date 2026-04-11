import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from utils.logger import log

from models.ddpm.diffusion import GaussianDiffusion
from .base import BaseTrainer

class DiffusionTrainer(BaseTrainer):
    """DDPM 训练器。

    这里把“网络结构”和“扩散过程”拆开：
    - `self.model` 只负责预测噪声
    - `self.diffusion` 负责 q_sample / p_sample 这些时间步逻辑
    """

    def __init__(self, config, train_loader, val_loader=None):
        diff_cfg = config.get("diffusion", {})
        self.diffusion = GaussianDiffusion(
            timesteps=diff_cfg.get("timesteps", 1000),
            beta_schedule=diff_cfg.get("schedule", "linear")
        )
        super().__init__(config, train_loader, val_loader)

    def _monitor_name(self):
        return "val_noise_loss"

    def _monitor_display_name(self):
        return "Val Noise Loss"

    def _monitor_mode(self):
        return "min"

    def _build_criterion(self):
        return F.mse_loss

    def _compute_noise_prediction_loss(self, images):
        batch_size = images.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
        noise = torch.randn_like(images)
        x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
        predicted_noise = self.model(x_noisy, t)
        return self.criterion(predicted_noise, noise)

    @torch.no_grad()
    def _save_samples(self, epoch):
        img_size = self.cfg["data"].get("image_size", 28)
        channels = self.cfg["model"]["params"].get("in_channels", 1)
        num_samples = self.cfg.get("diffusion", {}).get("sample_batch_size", 16)
        nrow = min(4, num_samples)

        shape = (num_samples, channels, img_size, img_size)
        img = torch.randn(shape, device=self.device)

        for step in reversed(range(self.diffusion.timesteps)):
            t = torch.full((shape[0],), step, device=self.device, dtype=torch.long)
            img = self.diffusion.p_sample(self.model, img, t)

        sample_path = self.exp_dir / f"sample_epoch_{epoch+1}.png"
        value_range = self.cfg.get("data", {}).get("value_range", "minus_one_one")
        if value_range == "minus_one_one":
            save_image((img + 1.0) * 0.5, sample_path, nrow=nrow)
        else:
            save_image(img.clamp(0.0, 1.0), sample_path, nrow=nrow)
        log(f"Saved diffusion samples to {sample_path}")

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        log_interval = max(self.cfg.get("train", {}).get("log_interval", 50), 1)

        for batch_idx, (images, _) in enumerate(self.train_loader, start=1):
            images = images.to(self.device)
            loss = self._compute_noise_prediction_loss(images)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = images.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            if batch_idx == 1 or batch_idx % log_interval == 0 or batch_idx == len(self.train_loader):
                log(
                    f"Epoch [{epoch+1}] Batch [{batch_idx}/{len(self.train_loader)}] "
                    f"Noise MSE: {loss.item():.6f}"
                )

        avg_loss = total_loss / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Train Noise Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        if self.val_loader is not None:
            for images, _ in self.val_loader:
                images = images.to(self.device)
                loss = self._compute_noise_prediction_loss(images)
                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                total_samples += batch_size

        avg_loss = None
        if total_samples > 0:
            avg_loss = total_loss / total_samples
            log(f"Epoch [{epoch+1}] Val Noise Loss: {avg_loss:.6f}")

        self._save_samples(epoch)
        return avg_loss
