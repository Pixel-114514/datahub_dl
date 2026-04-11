from models.ddpm.unet import UNetModel


class SR3UNet(UNetModel):
    """教学版 SR3 条件 UNet。

    SR3 的关键桥梁作用不是“换一套完全不同的网络”，而是让学员看到：
    同一个扩散 UNet，加上条件输入，就能从无条件生成走到条件恢复。
    """

    def __init__(self, in_channels=2, out_channels=1, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
        )
