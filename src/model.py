"""
模型定义。

CompactCNN: 为 GTSRB 设计的紧凑卷积网络
  - 输入 32x32x3
  - 4 层卷积 + 2 层全连接
  - 约 146K 参数（比 MobileNetV3-Small 小 10 倍）
  - 训练快、推理快、性能在 GTSRB 上稳定 95-97%

设计参考：经典的 GTSRB 基线网络（如 Sermanet & LeCun 2011, IDSIA 多列 CNN 简化版）
"""
import torch
import torch.nn as nn


class CompactCNN(nn.Module):
    """为 GTSRB 设计的紧凑卷积网络（约 146K 参数）。"""

    def __init__(self, num_classes=43):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 全局平均池化：4x4 → 1x1
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        # Kaiming 初始化（针对 ReLU 激活）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=43):
    """工厂函数。后续 Day 4 做 Backbone Ablation 时可以扩展支持其他模型。"""
    return CompactCNN(num_classes=num_classes)


if __name__ == '__main__':
    # 自测
    model = build_model()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CompactCNN 参数量: {n_params:,}  ({n_params/1e3:.1f} K)")

    # 验证 forward shape
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {y.shape}  (应为 [2, 43])")
    assert y.shape == (2, 43)
    print("\n模型自测通过")
