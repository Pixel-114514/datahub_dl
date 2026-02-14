import argparse
import yaml
from data.dataloader import get_dataloader
from trainer.base import BaseTrainer
import torch
from matplotlib import pyplot as plt
from inference.inference import predict_batch, predict_single_image
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Simple DL Project Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="./configs/cnn.yaml",
        help="Path to the YAML config file (default: configs/cnn.yaml)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Override device: cuda / cpu / mps (optional)"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="Path to image for prediction. If 'batch', predict a random batch from test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint for prediction"
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
        config["device"] = args.device

    # 如果是预测模式
    if args.predict:
        if not args.checkpoint:
            print("Error: --checkpoint is required for prediction mode")
            return
            
        # 只需要测试集用于 batch 预测，或者根本不需要 dataloader (单图预测)
        _, test_loader = get_dataloader(config)
        
        # 初始化 Trainer 来构建模型（复用 BaseTrainer 的构建逻辑）
        trainer = BaseTrainer(config=config, train_loader=None, val_loader=test_loader)
        
        # 加载权重
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        
        if args.predict == 'batch':
            print("Predicting a random batch from test set...")
            predict_batch(trainer.model, test_loader, trainer.device)
        else:
            print(f"Predicting single image: {args.predict}")
            predict_single_image(trainer.model, args.predict, trainer.device)
            
        return

    # 训练模式
    # 数据加载器
    train_loader, test_loader = get_dataloader(config)

    # 训练器
    trainer = BaseTrainer(
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