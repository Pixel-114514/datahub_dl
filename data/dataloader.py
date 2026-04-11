import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from data.sr_dataset import SyntheticSuperResolutionDataset
from data.transforms import build_transform


DATASET_REGISTRY = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "fashionmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

def _build_dataset(dataset_name, root, train, transform, download):
    dataset_key = dataset_name.lower()
    if dataset_key not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Supported datasets: {list(DATASET_REGISTRY.keys())}"
        )

    dataset_class = DATASET_REGISTRY[dataset_key]
    return dataset_class(
        root=root,
        train=train,
        download=download,
        transform=transform,
    )


def _maybe_subset(dataset, max_samples):
    if not max_samples or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _infer_task(config):
    trainer_name = config.get("trainer_name", "base")
    if trainer_name in {"sr", "resshift"}:
        return "super_resolution"
    return "standard"


def get_dataloader(config):
    data_cfg = config.get("data", {})
    train_cfg = config.get("train", {})
    task = data_cfg.get("task", _infer_task(config))
    transform = build_transform(data_cfg)

    root = data_cfg.get("root", "./data")
    dataset_name = data_cfg.get("dataset", "mnist")
    download = data_cfg.get("download", True)

    train_dataset = _build_dataset(dataset_name, root, True, transform, download)
    test_dataset = _build_dataset(dataset_name, root, False, transform, download)

    if task == "super_resolution":
        sr_kwargs = {
            "scale_factor": data_cfg.get("scale_factor", 2),
            "downsample_mode": data_cfg.get("downsample_mode", "bicubic"),
            "upsample_mode": data_cfg.get("upsample_mode", "bicubic"),
            "noise_std": data_cfg.get("noise_std", 0.0),
            "value_range": data_cfg.get("value_range", "zero_one"),
        }
        train_dataset = SyntheticSuperResolutionDataset(train_dataset, **sr_kwargs)
        test_dataset = SyntheticSuperResolutionDataset(test_dataset, **sr_kwargs)

    train_dataset = _maybe_subset(train_dataset, data_cfg.get("max_train_samples"))
    test_dataset = _maybe_subset(test_dataset, data_cfg.get("max_val_samples"))

    num_workers = data_cfg.get("num_workers", 0)
    loader_kwargs = {
        "batch_size": train_cfg.get("batch_size", 64),
        "num_workers": num_workers,
        "pin_memory": config.get("device", "cpu") == "cuda" and torch.cuda.is_available(),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, test_loader
