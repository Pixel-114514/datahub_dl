import torch
import torch.nn as nn
from utils.logger import log
from .base import BaseTrainer   # 按你实际路径修改


class VAETrainer(BaseTrainer):

    # ======================
    # 覆盖 loss
    # ======================
    def _build_criterion(self):

        def vae_loss(x_recon, x, mu, logvar):
            x = x.view(x.size(0), -1)
            x_recon = x_recon.view(x.size(0), -1)

            recon_loss = nn.functional.binary_cross_entropy(
                x_recon, x, reduction="sum"
            )

            kl_loss = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            return recon_loss + kl_loss, recon_loss, kl_loss

        return vae_loss

    # ======================
    # 训练
    # ======================
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        total_samples = 0

        for x, _ in self.train_loader:
            x = x.to(self.device)

            x_recon, mu, logvar = self.model(x)
            loss, recon_loss, kl_loss = self.criterion(
                x_recon, x, mu, logvar
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        avg_recon = total_recon / total_samples
        avg_kl = total_kl / total_samples

        log(
            f"Epoch [{epoch+1}] "
            f"Loss: {avg_loss:.4f} | "
            f"Recon: {avg_recon:.4f} | "
            f"KL: {avg_kl:.4f}"
        )

        return avg_loss

    def save_checkpoint(self, epoch, is_best=False):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "cfg": self.cfg,
        }

        last_path = self.exp_dir / "last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.exp_dir / "best.pth"
            torch.save(checkpoint, best_path)
            log(
                f"New best model saved! "
                f"Val Loss: {self.best_metric:.4f} "
                f"@ epoch {epoch+1}"
            )

        log(f"Checkpoint saved: {last_path}")
    # ======================
    # 验证
    # ======================
    @torch.no_grad()
    def evaluate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for x, _ in self.val_loader:
            x = x.to(self.device)

            x_recon, mu, logvar = self.model(x)
            loss, _, _ = self.criterion(
                x_recon, x, mu, logvar
            )

            total_loss += loss.item()
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        log(f"Epoch [{epoch+1}] Val Loss: {avg_loss:.4f}")
        return avg_loss

    # ======================
    # 覆盖 fit (loss 越小越好)
    # ======================
    def fit(self):
        epochs = self.cfg["train"]["epochs"]
        save_interval = self.cfg.get("save_interval", 1)

        self.best_metric = float("inf")

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_metric = self.evaluate(epoch)

            if val_metric is not None and val_metric < self.best_metric:
                self.best_metric = val_metric
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            elif (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)

        log(
            f"Training finished. "
            f"Best Val Loss: {self.best_metric:.4f} "
            f"@ epoch {self.best_epoch+1}"
        )