from torchvision import transforms


def build_transform(data_cfg):
    """根据配置拼装图像预处理流程。

    目前项目故意保持简单，只保留教学里最常用的几步：
    1. Resize 到统一尺寸
    2. ToTensor 转成张量
    3. 按任务需要切换到 [0, 1] 或 [-1, 1] 数值范围
    """

    transform_steps = []
    image_size = data_cfg.get("image_size")
    if image_size:
        transform_steps.append(transforms.Resize((image_size, image_size)))

    transform_steps.append(transforms.ToTensor())

    value_range = data_cfg.get("value_range", "zero_one")
    if value_range == "minus_one_one":
        transform_steps.append(transforms.Lambda(lambda x: x * 2.0 - 1.0))

    return transforms.Compose(transform_steps)
