import torch
import torch.nn as nn
import time  # 导入时间模块


class ResidualBlock1(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock1, self).__init__()
        out_channels = in_channels * 2  # 输出通道是输入通道的两倍
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # shortcut 用于调整输入形状
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x, original):
        # original是原始特征，x是增强特征
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += self.shortcut(original)
        out = self.relu(out)
        return out


class StackedResidualBlocks1(nn.Module):
    def __init__(self, in_channels, num_blocks):
        super(StackedResidualBlocks1, self).__init__()
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResidualBlock1(in_channels))
            in_channels *= 2  # 每个块后通道数翻倍

    def forward(self, x):
        original = x  # 保存原始特征
        for block in self.blocks:
            x = block(x, original)
            original = x  # 更新原始特征为当前输出
        return x


if __name__ == "__main__":
    # 定义输入
    input_tensor = torch.randn(8, 64, 16, 32, 32)  # 假设的输入，形状为 (批量大小, 通道数, 深度, 高度, 宽度)

    # 创建堆叠的残差块模型
    model = StackedResidualBlocks1(in_channels=64, num_blocks=2)  # 堆叠 3 次

    start_time = time.time()  # 开始计时
    output = model(input_tensor)  # 前向传播
    end_time = time.time()  # 结束计时

    print("Output shape:", output.shape)  # 检查输出形状
    print("Time taken for forward pass: {:.4f} seconds".format(end_time - start_time))