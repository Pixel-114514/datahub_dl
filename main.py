import argparse
import yaml
import torch
from data.dataloader import get_dataloader

# 注意：这里我们直接导入 registry 字典，而不是具体的 Trainer 类
from trainer import TRAINER_REGISTRY

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
        config["device"] = args.device

    # 数据加载器
    train_loader, test_loader = get_dataloader(config)

    # --- 注册机制核心代码开始 ---
    
    # 1. 从配置中获取 trainer 名称，默认为 'base'
    trainer_name = config.get("trainer_name", "base")
    
    # 2. 从注册表中查找对应的类
    if trainer_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Trainer '{trainer_name}' 未注册。"
            f"可用的 Trainers: {list(TRAINER_REGISTRY.keys())}"
        )
    
    trainer_class = TRAINER_REGISTRY[trainer_name]
    print(f"Initializing trainer: {trainer_name} ({trainer_class.__name__})")

    # 3. 实例化训练器
    trainer = trainer_class(
        config=config,
        train_loader=train_loader,
        val_loader=test_loader,
    )
    
    # --- 注册机制核心代码结束 ---

    trainer.fit()


if __name__ == "__main__":
    # 打印 GPU 信息
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 是否可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA 版本:", torch.version.cuda)
        print("GPU 数量:", torch.cuda.device_count())
        print("当前 GPU 名称:", torch.cuda.get_device_name(0))
        print("当前默认设备:", torch.cuda.current_device())
    else:
        print("没有检测到可用 GPU")

    main()
