import torch
import torch.nn as nn
import time  # 导入时间模块


class ResidualBlock2(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock2, self).__init__()
        out_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 调整输入 x 的形状以匹配 out
        if out.shape[2:] != x.shape[2:]:
            # 使用插值方法调整形状
            x = nn.functional.interpolate(x, size=out.shape[2:], mode='trilinear', align_corners=False)

        # 检查形状是否匹配
        if out.shape != x.shape:
            raise ValueError(f"Shape mismatch: out shape {out.shape}, input shape {x.shape}")
        # 残差连接
        out += x
        out = self.relu(out)
        return out


class StackedResidualBlocks(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(StackedResidualBlocks, self).__init__()
        self.blocks = nn.ModuleList([ResidualBlock2(in_channels) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


if __name__ == "__main__":
    # 定义输入
    input_tensor = torch.randn(8, 64, 16, 32, 32)  # 假设的输入，形状为 (批量大小, 通道数, 深度, 高度, 宽度)

    # 创建堆叠的残差块模型
    model = StackedResidualBlocks(in_channels=64, num_blocks=3)  # 堆叠 3 次

    start_time = time.time()  # 开始计时
    output = model(input_tensor)  # 前向传播
    end_time = time.time()  # 结束计时

    print("Output shape:", output.shape)  # 检查输出形状
    print("Time taken for forward pass: {:.4f} seconds".format(end_time - start_time))
    