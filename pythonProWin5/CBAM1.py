# 封装好的代码接口
import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3, 4), keepdim=True)
        max_pool = torch.max(x, dim=2)[0]
        max_pool = torch.max(max_pool, dim=3)[0]
        max_pool = torch.max(max_pool, dim=2)[0]

        avg_out = self.fc2(torch.relu(self.fc1(avg_pool.view(avg_pool.size(0), -1))))
        max_out = self.fc2(torch.relu(self.fc1(max_pool.view(max_pool.size(0), -1))))

        out = torch.sigmoid(avg_out + max_out).view(x.size(0), -1, 1, 1, 1)
        return x * out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]

        concat = torch.cat([avg_pool, max_pool], dim=1)
        out = torch.sigmoid(self.conv(concat))
        return x * out

class CBAM(nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()


    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class CWT(nn.Module):
    def __init__(self):
        super(CWT, self).__init__()

        # 定义可学习的加权系数
        self.alpha1 = nn.Parameter(torch.tensor(0.1))  # 权重值初始值
        self.alpha2 = nn.Parameter(torch.tensor(0.3))
        self.alpha3 = nn.Parameter(torch.tensor(0.2))
        self.alpha4 = nn.Parameter(torch.tensor(0.4))

    def forward(self, F1_enhanced, F2_enhanced, F3_enhanced, F4_enhanced):
        # 计算每个增强特征图的标量权重
        w1 = torch.mean(F1_enhanced)  # 使用增强特征计算权重
        w2 = torch.mean(F2_enhanced)
        w3 = torch.mean(F3_enhanced)
        w4 = torch.mean(F4_enhanced)

        # 计算加权平均权重
        weighted_sum = (self.alpha1 * w1 + self.alpha2 * w2 +
                        self.alpha3 * w3 + self.alpha4 * w4)
        final_weight = torch.sigmoid(weighted_sum / (self.alpha1 + self.alpha2 + self.alpha3 + self.alpha4))  # 归一化

        # 返回 final_weight
        return final_weight