import torch
import torch.nn as nn


class SimpleCNN(nn.Module): # 继承于nn.Module
    """
    一个简单但效果不错的CNN，适合MNIST手写数字分类
    目标：参数少、训练快、测试准确率轻松99%+
    """
    def __init__(
        self,
        num_classes: int = 10,
        dropout_rate: float = 0.25,
        activation: str = "gelu",      # gelu / relu / silu
    ):
        super().__init__()
        
        # 选择激活函数
        if activation.lower() == "gelu":
            act = nn.GELU()
        elif activation.lower() == "silu":
            act = nn.SiLU()
        else:
            act = nn.ReLU()
        # 组织网络提取特征
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),     # 28→28
            nn.BatchNorm2d(32),
            act,
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),          # 28→14
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.MaxPool2d(kernel_size=2, stride=2),          # 14→7
            
            # 全局平均池化（比flatten更现代，参数更少）
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # 使用线性全连接层作为分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),
            act,
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # 简单权重初始化
        self._init_weights()
    

    def _init_weights(self):
        for m in self.modules():
            # 卷积层
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 处理偏置
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # 归一层
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            # 全连接层
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, 28, 28)
        x = self.features(x)          # → (B, 64, 1, 1)
        x = self.classifier(x)        # → (B, 10)
        return x


# ────────────────────────────────────────────────
# 使用示例
# ────────────────────────────────────────────────

if __name__ == "__main__":
    model = SimpleCNN(num_classes=10, dropout_rate=0.3, activation="gelu")
    
    # 测试输入形状
    x = torch.randn(64, 1, 28, 28)
    out = model(x)
    print("Output shape:", out.shape)      # torch.Size([64, 10])
    print("\nModel:\n", model)
    
    # 参数量统计（大概几万参数，很轻量）
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {total_params:,}")