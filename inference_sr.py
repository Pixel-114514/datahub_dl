"""超分模型推理脚本。

支持 SRResNet / SR3 / ResShift 三种模型的推理和可视化。
用法：
    python inference_sr.py --ckpt checkpoints/resshift_mnist_toy/best.pth
    python inference_sr.py --ckpt checkpoints/sr3_mnist_toy/best.pth --device cpu
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import models
from data.sr_dataset import SyntheticSuperResolutionDataset
from utils.metrics import calculate_psnr


def load_trainer_and_config(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["cfg"]

    trainer_name = cfg.get("trainer_name", "base")
    model_name = cfg["model"]["name"]
    model_params = cfg["model"]["params"]

    model = models.MODEL_REGISTRY[model_name](**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model: {model_name}, Trainer: {trainer_name}")
    print(f"Trained epochs: {checkpoint['epoch'] + 1}")

    return model, cfg, trainer_name, device


def build_sr_dataloader(cfg):
    data_cfg = cfg.get("data", {})
    transform_steps = []
    image_size = data_cfg.get("image_size")
    if image_size:
        transform_steps.append(transforms.Resize((image_size, image_size)))
    transform_steps.append(transforms.ToTensor())
    value_range = data_cfg.get("value_range", "zero_one")
    if value_range == "minus_one_one":
        transform_steps.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))
    transform = transforms.Compose(transform_steps)

    test_dataset = datasets.MNIST(
        root=data_cfg.get("root", "./data"),
        train=False,
        download=True,
        transform=transform,
    )

    sr_kwargs = {
        "scale_factor": data_cfg.get("scale_factor", 2),
        "downsample_mode": data_cfg.get("downsample_mode", "bicubic"),
        "upsample_mode": data_cfg.get("upsample_mode", "bicubic"),
        "noise_std": data_cfg.get("noise_std", 0.0),
        "value_range": value_range,
    }
    test_dataset = SyntheticSuperResolutionDataset(test_dataset, **sr_kwargs)

    return DataLoader(test_dataset, batch_size=8, shuffle=True)


def get_clamp_range(cfg):
    value_range = cfg.get("data", {}).get("value_range", "zero_one")
    if value_range == "minus_one_one":
        return (-1.0, 1.0)
    return (0.0, 1.0)


def infer_sr(model, lr, trainer_name, cfg, device):
    """根据 trainer 类型选择推理方式。"""
    lr = lr.to(device)

    if trainer_name == "sr":
        return model(lr)

    if trainer_name == "sr3":
        from models.ddpm.diffusion import GaussianDiffusion

        diff_cfg = cfg.get("diffusion", {})
        diffusion = GaussianDiffusion(
            timesteps=diff_cfg.get("timesteps", 100),
            beta_schedule=diff_cfg.get("schedule", "linear"),
        )
        clamp_range = get_clamp_range(cfg)

        current = torch.randn_like(lr)
        batch_size = lr.shape[0]
        for step in reversed(range(diffusion.timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            predicted_noise = model(torch.cat([current, lr], dim=1), t)
            current = diffusion.p_sample_from_pred_noise(
                current, t, predicted_noise, clip_denoised_range=clamp_range
            )
        return current.clamp(*clamp_range)

    if trainer_name == "resshift":
        from models.resshift import ResidualShiftScheduler

        resshift_cfg = cfg.get("resshift", {})
        scheduler = ResidualShiftScheduler(
            timesteps=resshift_cfg.get("timesteps", 15),
            noise_level=resshift_cfg.get("noise_level", 0.2),
            schedule=resshift_cfg.get("schedule", "linear"),
        )
        return scheduler.sample(model, lr, clamp_range=get_clamp_range(cfg))

    raise ValueError(f"Unsupported trainer for SR inference: {trainer_name}")


def save_comparison(lr, sr, hr, save_path, clamp_range):
    """保存三行对比图：低清 / 超分 / 高清。"""
    lr = lr.cpu().clamp(*clamp_range)
    sr = sr.cpu().clamp(*clamp_range)
    hr = hr.cpu().clamp(*clamp_range)

    num_images = lr.shape[0]
    fig, axes = plt.subplots(3, num_images, figsize=(num_images * 2, 6))

    for i in range(num_images):
        axes[0, i].imshow(lr[i, 0], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(sr[i, 0], cmap="gray")
        axes[1, i].axis("off")
        axes[2, i].imshow(hr[i, 0], cmap="gray")
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("LR Input", fontsize=10)
    axes[1, 0].set_ylabel("SR Output", fontsize=10)
    axes[2, 0].set_ylabel("HR Ground Truth", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Super-Resolution Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (best.pth or last.pth)")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="inference_results", help="Output directory")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, trainer_name, device = load_trainer_and_config(args.ckpt, device)
    loader = build_sr_dataloader(cfg)
    clamp_range = get_clamp_range(cfg)
    data_range = clamp_range[1] - clamp_range[0]

    lr, hr = next(iter(loader))
    lr, hr = lr.to(device), hr.to(device)

    with torch.no_grad():
        sr = infer_sr(model, lr, trainer_name, cfg, device)

    psnr = calculate_psnr(sr.clamp(*clamp_range), hr, data_range=data_range)
    print(f"PSNR: {psnr:.2f} dB")

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    save_comparison(lr, sr, hr, save_dir / "sr_comparison.png", clamp_range)


if __name__ == "__main__":
    main()
