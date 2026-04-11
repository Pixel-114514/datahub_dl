import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

import models
from utils.logger import log


class BaseTrainer:
    """教学项目里的通用训练骨架。

    入口层只负责读取配置和选择 trainer，真正的训练生命周期都在这里：
    1. 构建设备、模型、loss、优化器
    2. 管理实验目录和 checkpoint
    3. 提供统一的 fit / evaluate / save 流程

    各个具体任务只需要覆写少量 hook，就能把“自己的训练逻辑”插进来。
    """

    def __init__(self, config, train_loader, val_loader=None):
        self.cfg = config
        self.device = self._build_device()

        self.train_loader = train_loader
        self.val_loader = val_loader

        self._MODEL_REGISTRY = models.MODEL_REGISTRY
        self.model = self._build_model().to(self.device)
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()

        self.monitor_name = self._monitor_name()
        self.monitor_display_name = self._monitor_display_name()
        self.monitor_mode = self._monitor_mode()
        self.monitor_unit = self._monitor_unit()
        self.best_metric = self._initial_best_metric()
        self.best_epoch = -1
        self.start_epoch = 0

        self.save_dir = Path(self.cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if "exp_name" not in self.cfg:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.cfg["model"]["name"]
            self.cfg["exp_name"] = f"{model_name}_{timestamp}"

        self.exp_dir = self.save_dir / self.cfg["exp_name"]
        self.exp_dir.mkdir(exist_ok=True)

        self._save_config()

    # =====================
    # build components
    # =====================

    def _build_device(self):
        requested_device = self.cfg.get("device", "cpu")
        print(f"Using device: {requested_device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if requested_device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")

        if (
            requested_device == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")

        return torch.device("cpu")

    def _build_model(self):
        model_cfg = self.cfg["model"]
        name = model_cfg["name"]
        
        if name not in self._MODEL_REGISTRY:
            raise ValueError(f"Unknown model name: {name}")

        params = model_cfg.get("params", {}) 
        return self._MODEL_REGISTRY[name](**params)

    def _build_criterion(self):
        return nn.CrossEntropyLoss()

    def _build_optimizer(self):
        return optim.Adam(
            self.model.parameters(),
            lr=self.cfg["train"]["lr"],
        )

    def _monitor_name(self):
        return "val_acc"

    def _monitor_display_name(self):
        return "Val Acc"

    def _monitor_mode(self):
        return "max"

    def _monitor_unit(self):
        return ""

    def _initial_best_metric(self):
        if self.monitor_mode == "max":
            return float("-inf")
        if self.monitor_mode == "min":
            return float("inf")
        raise ValueError(f"Unsupported monitor mode: {self.monitor_mode}")

    def _is_improvement(self, metric):
        if self.monitor_mode == "max":
            return metric > self.best_metric
        return metric < self.best_metric

    def _format_metric(self, value):
        if value is None:
            return "N/A"
        suffix = f" {self.monitor_unit}" if self.monitor_unit else ""
        return f"{value:.4f}{suffix}"

    # =====================
    # 保存相关
    # =====================

    def _save_config(self):
        """保存当前配置，便于复现"""
        config_path = self.exp_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.cfg, f, indent=2, ensure_ascii=False)
        log(f"Config saved to: {config_path}")

    def save_checkpoint(self, epoch, is_best=False):
        """保存模型权重 + 优化器状态 + epoch 等信息"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric if self.best_epoch >= 0 else None,
            "best_epoch": self.best_epoch,
            "monitor_name": self.monitor_name,
            "monitor_mode": self.monitor_mode,
            "cfg": self.cfg,
        }

        last_path = self.exp_dir / "last.pth"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.exp_dir / "best.pth"
            torch.save(checkpoint, best_path)
            log(
                f"New best model saved! "
                f"{self.monitor_display_name}: {self._format_metric(self.best_metric)} "
                f"@ epoch {epoch+1}"
            )

        log(f"Checkpoint saved: {last_path}")

    # =====================
    # train / eval
    # =====================

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            loss = self.criterion(logits, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

        avg_loss = total_loss / total_samples
        log(f"Epoch [{epoch+1}] Train Loss: {avg_loss:.4f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, epoch):
        if self.val_loader is None:
            return None

        self.model.eval()
        correct, total = 0, 0

        for x, y in self.val_loader:
            x = x.to(self.device)
            y = y.to(self.device)

            logits = self.model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        acc = correct / total
        log(f"Epoch [{epoch+1}] Val Acc: {acc:.4f}")
        return acc

    def fit(self):
        epochs = self.cfg["train"]["epochs"]
        save_interval = max(self.cfg.get("save_interval", 1), 1)

        for epoch in range(self.start_epoch, epochs):
            self.train_one_epoch(epoch)
            val_metric = self.evaluate(epoch)

            is_best = val_metric is not None and self._is_improvement(val_metric)
            if is_best:
                self.best_metric = val_metric
                self.best_epoch = epoch

            should_save = is_best or (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs
            if should_save:
                self.save_checkpoint(epoch, is_best=is_best)

        if self.best_epoch >= 0:
            log(
                f"Training finished. Best {self.monitor_display_name}: "
                f"{self._format_metric(self.best_metric)} @ epoch {self.best_epoch+1}"
            )
        else:
            log("Training finished. No validation metric was tracked.")

        log(f"Experiment directory: {self.exp_dir}")
