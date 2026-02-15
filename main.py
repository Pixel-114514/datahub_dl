import argparse
import yaml
from data.dataloader import get_dataloader
from trainer.base import BaseTrainer
from trainer.vae import VAETrainer
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Simple DL Project Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/resnet.yaml",
        help="Path to the YAML config file (default: configs/resnet.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device: cuda / cpu / mps (optional)"
    )

    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 读取配置文件
    config_path = args.config
    print(f"Loading config from: {config_path}")
    
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 {config_path} 不存在")
        return
    except Exception as e:
        print(f"读取配置文件失败：{e}")
        return

    # 可选：命令行参数覆盖 config 中的值
    if args.device:
        config["device"] = args.device  # 假设 config 里有 device 字段

    # 数据加载器
    train_loader, test_loader = get_dataloader(config)

    # 训练器
    trainer = VAETrainer(
        config=config,
        train_loader=train_loader,
        val_loader=test_loader,
    )

    trainer.fit()


if __name__ == "__main__":
    # 先打印 GPU 信息（保持你原来的 debug 代码）
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 版本 (torch 编译时用的):", torch.version.cuda)
        print("GPU 数量:", torch.cuda.device_count())
        print("当前 GPU 名称:", torch.cuda.get_device_name(0))
        print("当前默认设备:", torch.cuda.current_device())
    else:
        print("没有检测到可用 GPU")

    main()