import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((27, 5, 5))  # 输出尺寸统一为 [C, 27, 5, 5]

    def forward(self, feat1, feat2, feat3, feat4):
        # 输入形状： [B, C, D, H, W]
        pooled_feat1 = self.avg_pool(feat1)
        pooled_feat2 = self.avg_pool(feat2)
        pooled_feat3 = self.avg_pool(feat3)
        pooled_feat4 = self.avg_pool(feat4)

        # 拼接：维度必须一致（除了拼接维）
        fused = torch.cat((pooled_feat1, pooled_feat2, pooled_feat3, pooled_feat4), dim=1)  # 在 channel 拼接

        return fused
