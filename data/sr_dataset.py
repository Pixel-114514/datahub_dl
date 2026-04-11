import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SyntheticSuperResolutionDataset(Dataset):
    """把普通分类数据集包装成超分训练对。

    对学员来说，这个类最值得理解的点是：
    原始数据集只有 `(image, label)`，但超分训练真正需要的是 `(lr_up, hr)`。
    这个包装层把“任务样本定义”从“原始数据集定义”里分离了出来。
    """

    def __init__(
        self,
        base_dataset,
        scale_factor=2,
        downsample_mode="bicubic",
        upsample_mode="bicubic",
        noise_std=0.0,
        value_range="zero_one",
    ):
        self.base_dataset = base_dataset
        self.scale_factor = scale_factor
        self.downsample_mode = downsample_mode
        self.upsample_mode = upsample_mode
        self.noise_std = noise_std
        self.value_range = value_range

    def __len__(self):
        return len(self.base_dataset)

    def _clamp(self, image):
        if self.value_range == "minus_one_one":
            return image.clamp(-1.0, 1.0)
        return image.clamp(0.0, 1.0)

    def _resize(self, image, size, mode):
        kwargs = {}
        if mode in {"bilinear", "bicubic"}:
            kwargs["align_corners"] = False
        return F.interpolate(image, size=size, mode=mode, **kwargs)

    def __getitem__(self, index):
        hr, _ = self.base_dataset[index]
        hr = self._clamp(hr)
        hr_size = hr.shape[-2:]
        lr_size = tuple(max(1, dim // self.scale_factor) for dim in hr_size)

        lr = self._resize(hr.unsqueeze(0), lr_size, self.downsample_mode).squeeze(0)
        if self.noise_std > 0:
            lr = lr + torch.randn_like(lr) * self.noise_std
        lr = self._clamp(lr)

        lr_up = self._resize(lr.unsqueeze(0), hr_size, self.upsample_mode).squeeze(0)
        lr_up = self._clamp(lr_up)
        return lr_up, hr
