"""三种超分方法对比脚本。

加载 SRResNet / SR3 / ResShift 的 checkpoint，在同一组输入上做推理，
生成五行并排对比图（低清 / SRResNet / SR3 / ResShift / 高清），并打印各自的 PSNR。

用法：
    python compare_sr.py \
        --sr_ckpt checkpoints/srresnet_mnist/best.pth \
        --sr3_ckpt checkpoints/sr3_mnist_toy/best.pth \
        --resshift_ckpt checkpoints/resshift_mnist_toy/best.pth

    # 只对比其中两种
    python compare_sr.py \
        --sr3_ckpt checkpoints/sr3_mnist_toy/best.pth \
        --resshift_ckpt checkpoints/resshift_mnist_toy/best.pth

    # 指定设备和输出目录
    python compare_sr.py \
        --sr_ckpt checkpoints/srresnet_mnist/best.pth \
        --sr3_ckpt checkpoints/sr3_mnist_toy/best.pth \
        --resshift_ckpt checkpoints/resshift_mnist_toy/best.pth \
        --device cpu --save_dir results --num_images 8
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from inference_sr import (
    build_sr_dataloader,
    get_clamp_range,
    infer_sr,
    load_trainer_and_config,
)
from utils.metrics import calculate_psnr


def load_model(ckpt_path, device):
    """加载模型，返回 (model, cfg, trainer_name)。"""
    model, cfg, trainer_name, _ = load_trainer_and_config(ckpt_path, device)
    return model, cfg, trainer_name


def run_inference(model, lr, hr, trainer_name, cfg, device, clamp_range):
    """跑推理并返回 (sr_images, psnr)。"""
    with torch.no_grad():
        sr = infer_sr(model, lr, trainer_name, cfg, device)
    sr = sr.clamp(*clamp_range)
    data_range = clamp_range[1] - clamp_range[0]
    psnr = calculate_psnr(sr, hr, data_range=data_range)
    return sr, psnr


def save_comparison(lr, results, hr, save_path, clamp_range, num_images):
    """保存五行对比图：LR / SRResNet / SR3 / ResShift / HR。

    results: list of (name, sr_images, psnr)
    """
    lr = lr.cpu().clamp(*clamp_range)
    hr = hr.cpu().clamp(*clamp_range)

    n_rows = 2 + len(results)  # LR + methods + HR
    fig, axes = plt.subplots(n_rows, num_images, figsize=(num_images * 2, n_rows * 2))

    # 确保 axes 是二维的
    if num_images == 1:
        axes = axes[:, None]

    # 第一行：低清输入
    for i in range(num_images):
        axes[0, i].imshow(lr[i, 0], cmap="gray")
        axes[0, i].axis("off")
    axes[0, 0].set_ylabel("LR Input", fontsize=10)

    # 中间行：各方法输出
    for row_idx, (name, sr, psnr) in enumerate(results):
        sr = sr.cpu().clamp(*clamp_range)
        for i in range(num_images):
            axes[row_idx + 1, i].imshow(sr[i, 0], cmap="gray")
            axes[row_idx + 1, i].axis("off")
        axes[row_idx + 1, 0].set_ylabel(f"{name}\n{psnr:.1f}dB", fontsize=10)

    # 最后一行：高清真值
    for i in range(num_images):
        axes[-1, i].imshow(hr[i, 0], cmap="gray")
        axes[-1, i].axis("off")
    axes[-1, 0].set_ylabel("HR GT", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare SRResNet / SR3 / ResShift side by side"
    )
    parser.add_argument("--sr_ckpt", type=str, default=None, help="SRResNet checkpoint path")
    parser.add_argument("--sr3_ckpt", type=str, default=None, help="SR3 checkpoint path")
    parser.add_argument("--resshift_ckpt", type=str, default=None, help="ResShift checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    parser.add_argument("--save_dir", type=str, default="compare_results", help="Output directory")
    parser.add_argument("--num_images", type=int, default=6, help="Number of images to compare")
    args = parser.parse_args()

    ckpts = {}
    if args.sr_ckpt:
        ckpts["SRResNet"] = args.sr_ckpt
    if args.sr3_ckpt:
        ckpts["SR3"] = args.sr3_ckpt
    if args.resshift_ckpt:
        ckpts["ResShift"] = args.resshift_ckpt

    if not ckpts:
        parser.error("至少提供一个 checkpoint 路径 (--sr_ckpt / --sr3_ckpt / --resshift_ckpt)")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 加载所有模型
    models_info = []
    for name, ckpt_path in ckpts.items():
        print(f"\nLoading {name}...")
        model, cfg, trainer_name = load_model(ckpt_path, device)
        models_info.append((name, model, trainer_name, cfg))

    # 用第一个模型的配置构建 dataloader
    loader = build_sr_dataloader(models_info[0][3])
    clamp_range = get_clamp_range(models_info[0][3])

    # 取一批数据
    lr, hr = next(iter(loader))
    lr, hr = lr.to(device), hr.to(device)
    num_images = min(args.num_images, lr.shape[0])
    lr, hr = lr[:num_images], hr[:num_images]

    # 跑推理
    results = []
    for name, model, trainer_name, cfg in models_info:
        print(f"Running {name} inference...")
        sr, psnr = run_inference(model, lr, hr, trainer_name, cfg, device, clamp_range)
        results.append((name, sr, psnr))
        print(f"  {name} PSNR: {psnr:.2f} dB")

    # 保存对比图
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / "sr_comparison.png"
    save_comparison(lr, results, hr, save_path, clamp_range, num_images)

    # 打印汇总
    print(f"\n{'='*40}")
    print(f"{'Method':<12} {'PSNR (dB)':>10}")
    print(f"{'-'*40}")
    for name, _, psnr in results:
        print(f"{name:<12} {psnr:>10.2f}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
