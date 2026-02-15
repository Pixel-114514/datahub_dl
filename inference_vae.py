import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
import models


# ==============================
# 1. 加载模型
# ==============================
def load_model(ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)
    cfg = checkpoint["cfg"]

    model_name = cfg["model"]["name"]
    model_params = cfg["model"]["params"]

    model = models.MODEL_REGISTRY[model_name](**model_params)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from: {ckpt_path}")
    print(f"Model: {model_name}")
    print(f"Epoch: {checkpoint['epoch']+1}")

    return model, cfg, device


# ==============================
# 2. 重构测试图片
# ==============================
def reconstruct(model, device, save_dir, num_images=16):
    transform = transforms.ToTensor()
    dataset = datasets.MNIST("./data", train=False, transform=transform)
    loader = DataLoader(dataset, batch_size=num_images, shuffle=True)

    x, _ = next(iter(loader))
    x = x.to(device)

    with torch.no_grad():
        x_recon, _, _ = model(x)

    x = x.cpu().view(-1, 28, 28)
    x_recon = x_recon.cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(2, num_images, figsize=(num_images, 3))

    for i in range(num_images):
        axes[0, i].imshow(x[i], cmap="gray")
        axes[0, i].axis("off")
        axes[1, i].imshow(x_recon[i], cmap="gray")
        axes[1, i].axis("off")

    plt.tight_layout()
    save_path = save_dir / "reconstruction.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Reconstruction saved to {save_path}")


# ==============================
# 3. 随机生成新样本
# ==============================
def generate(model, cfg, device, save_dir, num_samples=16):
    latent_dim = cfg["model"]["params"]["latent_dim"]

    with torch.no_grad():
        z = torch.randn(num_samples, latent_dim).to(device)
        samples = model.decode(z)

    samples = samples.cpu().view(-1, 28, 28)

    fig, axes = plt.subplots(4, 4, figsize=(6, 6))

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(samples[i], cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    save_path = save_dir / "generated.png"
    plt.savefig(save_path)
    plt.close()

    print(f"Generated samples saved to {save_path}")


# ==============================
# 4. 主函数
# ==============================
def main():
    ckpt_path = "checkpoints/vae_mnist/best.pth"
    save_dir = Path("inference_results")
    save_dir.mkdir(exist_ok=True)

    model, cfg, device = load_model(ckpt_path)

    reconstruct(model, device, save_dir)
    generate(model, cfg, device, save_dir)


if __name__ == "__main__":
    main()