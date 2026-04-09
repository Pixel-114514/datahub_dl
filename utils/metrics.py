import torch
import torch.nn.functional as F


def calculate_psnr(prediction, target, data_range=1.0, eps=1e-8):
    mse = F.mse_loss(prediction, target, reduction="none")
    mse = mse.flatten(1).mean(dim=1).clamp_min(eps)
    psnr = 10.0 * torch.log10((data_range ** 2) / mse)
    return psnr.mean().item()
