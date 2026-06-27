import torch
import torch.nn as nn
from CBAM1 import *
from XCA import *
from ResBlock1 import *
from ResBlock2 import *
from GCNet1 import *
from MyAvgnet import MyNetwork
import time


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        # 四种特征提取卷积
        self.AFE_conv1 = nn.Conv3d(1, 64, kernel_size=(3, 32, 32), stride=(3, 32, 32))
        self.AFE_conv1_bn = nn.BatchNorm3d(64)

        self.SFE_conv1 = nn.Conv3d(1, 64, kernel_size=(9, 1, 1), stride=1, dilation=(9, 1, 1))
        self.SFE_conv1_bn = nn.BatchNorm3d(64)

        self.EFE_H_conv1 = nn.Conv3d(1, 64, kernel_size=(3, 16, 1), stride=(3, 4, 1))
        self.EFE_H_conv1_bn = nn.BatchNorm3d(64)

        self.EFE_V_conv1 = nn.Conv3d(1, 64, kernel_size=(3, 1, 16), stride=(3, 1, 4))
        self.EFE_V_conv1_bn = nn.BatchNorm3d(64)

        # 统一的CBAM模块（增强拼接后的特征）
        self.CBAM = LinAngularXCA_CA()

        # 统一的GCNet处理
        self.GCNet = GCNet(in_channels=64 * 4, output_channels=1)

        # 金字塔池化融合
        self.avg_pool = MyNetwork()

        # 全连接预测部分
        self.FC1 = nn.Linear(4 * 27 * 5 * 5, 512)
        self.FC2 = nn.Linear(512, 128)
        self.FC3 = nn.Linear(128, 1)

        # 激活函数
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # 四个方向的卷积特征提取
        AFE_out = self.relu(self.AFE_conv1_bn(self.AFE_conv1(x)))     # [B, 64, 27, 1, 1]
        SFE_out = self.relu(self.SFE_conv1_bn(self.SFE_conv1(x)))     # [B, 64, 9, 32, 32]
        EFE_H_out = self.relu(self.EFE_H_conv1_bn(self.EFE_H_conv1(x)))  # [B, 64, 27, 5, 32]
        EFE_V_out = self.relu(self.EFE_V_conv1_bn(self.EFE_V_conv1(x)))  # [B, 64, 27, 32, 5]

        # 尺寸对齐（假设你希望用avg_pool实现对齐之前需要统一形状）
        # 使用avg_pool统一到 [B, 64, 27, 5, 5] 在池化融合之前
        # 你也可以自己加插值或conv3d实现对齐。这里我们使用自定义模块MyNetwork完成融合和对齐。

        # 拼接所有特征通道维度: [B, 64*4, ..., ..., ...]
        # 需要你确保MyNetwork能处理这种输入
        fused = self.avg_pool(AFE_out, SFE_out, EFE_H_out, EFE_V_out)
        # fused = torch.cat([AFE_out, SFE_out, EFE_H_out, EFE_V_out], dim=1)  # [B, 256, ?, ?, ?]

        # CBAM增强
        fused_cbam = self.CBAM(fused)  # [B, 256, ..., ..., ...]

        # 特征图处理
        fused_gcnet = self.GCNet(fused_cbam)  # [B, 1, 27, 5, 5] 假设输出固定

        # 特征池化融合
        FC = self.avg_pool(fused_gcnet, fused_gcnet, fused_gcnet, fused_gcnet)  # [B, 27, 5, 5]

        # 展平 + FC 层处理
        FC = FC.view(FC.size(0), -1)  # [B, 27*5*5]

        FC1 = self.relu(self.FC1(FC))
        FC2 = self.relu(self.FC2(FC1))
        Score = self.FC3(FC2)

        return Score


if __name__ == "__main__":
    model = Network()
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / (1024 * 1024)
    print(f"Total parameters: {total_params:,} ({total_params_m:.2f} MB)")

    data = torch.randn(8, 1, 81, 32, 32)
    start_time = time.time()
    score = model(data)
    end_time = time.time()
    print("Time taken for forward pass: {:.4f} seconds".format(end_time - start_time))
