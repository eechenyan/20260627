import torch.nn as nn
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
        # original是原始特征，x是增强特征 残差块要处理的对象
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += self.shortcut(original)
        out = self.relu(out)
        return out

        # 经过这些处理后，x会变成一个经过多层卷积、归一化和激活后的特征表示。最终的输出out是一个丰富的特征表示，
        # 它结合了原始特征的信息，这种设计使得网络能够更有效地学习复杂的模式，同时保持了输入信息的完整性。