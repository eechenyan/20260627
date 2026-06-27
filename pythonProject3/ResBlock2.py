import torch.nn as nn
class ResidualBlock2(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock2, self).__init__()
        # 输出通道数与输入通道数相同
        out_channels = in_channels

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        # shortcut 用于直接连接输入
        self.shortcut = nn.Sequential()

    def forward(self, x):
        # 残差块处理对象
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += x  # 直接连接输入
        out = self.relu(out)  # 激活函数
        return out
