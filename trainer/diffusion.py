from .base import BaseTrainer
from models.ddpm.diffusion import GaussianDiffusion
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

class DiffusionTrainer(BaseTrainer):
    def __init__(self, config, train_loader, val_loader=None):
        # 1. 初始化扩散调度器
        diff_cfg = config.get("diffusion", {})
        self.diffusion = GaussianDiffusion(
            timesteps=diff_cfg.get("timesteps", 1000),
            beta_schedule=diff_cfg.get("schedule", "linear")
        )
        # 2. 调用父类初始化（会触发 build_model, build_optimizer 等）
        super().__init__(config, train_loader, val_loader)

    def _build_criterion(self):
        # 扩散模型通常使用 MSE Loss 预测噪声
        return F.mse_loss

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(self.train_loader):
            images = images.to(self.device)
            batch_size = images.shape[0]
            
            # 采样随机时间步 t
            t = torch.randint(0, self.diffusion.timesteps, (batch_size,), device=self.device).long()
            
            # 生成噪声并注入
            noise = torch.randn_like(images)
            x_noisy = self.diffusion.q_sample(x_start=images, t=t, noise=noise)
            
            # 预测噪声
            predicted_noise = self.model(x_noisy, t)
            loss = self.criterion(noise, predicted_noise)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            print(f"Batch [{batch_idx+1}/{len(self.train_loader)}] Loss: {loss.item():.6f}")
            
        avg_loss = total_loss / len(self.train_loader)
        self.log(f"Epoch [{epoch+1}] Train Loss: {avg_loss:.6f}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, epoch):
        # 扩散模型的评估通常是生成样本查看效果
        self.model.eval()
        # 从 data 节点读取 image_size，如果不存在则默认 28
        img_size = self.cfg["data"].get("image_size", 28) 
        channels = self.cfg["model"]["params"].get("in_channels", 1)
        
        # 采样流程
        shape = (16, channels, img_size, img_size) # 生成16张图
        img = torch.randn(shape, device=self.device)
        
        for i in reversed(range(0, self.diffusion.timesteps)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            img = self.diffusion.p_sample(self.model, img, t)
        
        # 保存生成结果
        sample_path = self.exp_dir / f"sample_epoch_{epoch+1}.png"
        save_image((img + 1.0) * 0.5, sample_path, nrow=4)
        self.log(f"Saved samples to {sample_path}")
        
        # 返回一个负 Loss 作为 "Acc" 来触发 save_checkpoint
        return -1.0 

    def log(self, msg):
        # 适配你框架里的 log
        from utils.logger import log
        log(msg)