import torch
import torch.nn as nn
from torchvision.utils import save_image

from .base import BaseTrainer
from utils.logger import log
from utils.metrics import calculate_psnr


class BaseSRTrainer(BaseTrainer):
    def _clamp_range(self):
        value_range = self.cfg.get("data", {}).get("value_range", "zero_one")
        if value_range == "minus_one_one":
            return (-1.0, 1.0)
        return (0.0, 1.0)

    def _data_range(self):
        low, high = self._clamp_range()
        return high - low

    def _save_visuals(self, lr, sr, hr, epoch):
        sample_path = self.exp_dir / f"sr_epoch_{epoch+1}.png"
        images = torch.cat([lr, sr, hr], dim=0).cpu()
        save_kwargs = {"nrow": lr.shape[0]}
        if self._clamp_range() == (-1.0, 1.0):
            save_kwargs["normalize"] = True
            save_kwargs["value_range"] = (-1.0, 1.0)
        save_image(images, sample_path, **save_kwargs)
        log(f"Saved super-resolution samples to {sample_path}")

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_psnr": self.best_metric,
            "cfg": self.cfg,
        }

        last_path = self.exp_dir / "last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.exp_dir / "best.pth"
            torch.save(checkpoint, best_path)
            log(f"New best model saved! PSNR: {self.best_metric:.4f} dB @ epoch {epoch+1}")

        log(f"Checkpoint saved: {last_path}")

    def infer(self, lr):
        raise NotImplementedError

    @torch.no_grad()
    def evaluate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_psnr = 0.0
        total_samples = 0
        preview = None

        for batch_idx, (lr, hr) in enumerate(self.val_loader):
            lr = lr.to(self.device)
            hr = hr.to(self.device)
            sr = self.infer(lr).clamp(*self._clamp_range())

            batch_size = lr.size(0)
            batch_psnr = calculate_psnr(sr, hr, data_range=self._data_range())
            total_psnr += batch_psnr * batch_size
            total_samples += batch_size

            if batch_idx == 0:
                preview = (lr[:8], sr[:8], hr[:8])

        avg_psnr = total_psnr / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Val PSNR: {avg_psnr:.4f} dB")

        if preview is not None:
            self._save_visuals(*preview, epoch)

        return avg_psnr

    def fit(self):
        epochs = self.cfg["train"]["epochs"]
        save_interval = self.cfg.get("save_interval", 1)
        self.best_metric = float("-inf")

        for epoch in range(self.start_epoch, epochs):
            self.train_one_epoch(epoch)
            val_metric = self.evaluate(epoch)

            if val_metric is not None and val_metric > self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            elif (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)

        log(
            f"Training finished. Best Val PSNR: {self.best_metric:.4f} dB "
            f"@ epoch {self.best_epoch+1}"
        )
        log(f"Experiment directory: {self.exp_dir}")


class SuperResolutionTrainer(BaseSRTrainer):
    def _build_criterion(self):
        return nn.L1Loss()

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for lr, hr in self.train_loader:
            lr = lr.to(self.device)
            hr = hr.to(self.device)

            sr = self.model(lr)
            loss = self.criterion(sr, hr)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = lr.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        log(f"Epoch [{epoch+1}] Train L1: {avg_loss:.6f}")
        return avg_loss

    def infer(self, lr):
        return self.model(lr)
