import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import os
from datetime import datetime

from utils.logger import log
import models
from trainer.animator import Animator

class BaseTrainer:

    def __init__(self, config, train_loader, val_loader=None):
        self.cfg = config
        self.device = self._build_device()

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self._MODEL_REGISTRY = models.MODEL_REGISTRY
        self.model = self._build_model().to(self.device)
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()

        # 保存相关路径
        self.save_dir = Path(self.cfg.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_acc = -1.0
        self.best_epoch = -1
        self.start_epoch = 0

        # 自动创建实验名称（如果没有指定）
        if "exp_name" not in self.cfg:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.cfg["model"]["name"]
            self.cfg["exp_name"] = f"{model_name}_{timestamp}"
        
        self.exp_dir = self.save_dir / self.cfg["exp_name"]
        self.exp_dir.mkdir(exist_ok=True)

        # 保存一份配置（方便复现）
        self._save_config()

    # =====================
    # build components
    # =====================

    def _build_device(self):
        print(f"Using device: {self.cfg.get('device', 'cpu')}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        return torch.device(
            "cuda" if self.cfg.get("device") == "cuda" and torch.cuda.is_available()
            else "cpu"
        )

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
            "best_acc": self.best_acc,
            "cfg": self.cfg,  # 可选：存一份配置快照
        }

        # 常规保存（每个 epoch 或每隔 N 个 epoch）
        last_path = self.exp_dir / "last.pth"
        torch.save(checkpoint, last_path)
        
        if is_best:
            best_path = self.exp_dir / "best.pth"
            torch.save(checkpoint, best_path)
            log(f"New best model saved! Acc: {self.best_acc:.4f} @ epoch {epoch+1}")

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
        save_interval = self.cfg.get("save_interval", 1)  # 每多少 epoch 保存一次

        # 初始化 Animator
        animator = Animator(xlabel='epoch', xlim=[1, epochs], legend=['train loss', 'val acc'])

        for epoch in range(self.start_epoch, epochs):
            train_loss = self.train_one_epoch(epoch)
            val_acc = self.evaluate(epoch)

            # 更新可视化
            if val_acc is not None:
                animator.add(epoch + 1, (train_loss, val_acc))
            else:
                animator.add(epoch + 1, (train_loss, None))

            # 更新 best
            if val_acc is not None and val_acc > self.best_acc:
                self.best_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            elif (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)

        log(f"Training finished. Best Val Acc: {self.best_acc:.4f} @ epoch {self.best_epoch+1}")
        log(f"Experiment directory: {self.exp_dir}")



