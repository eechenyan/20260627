import torch
import torch.nn as nn


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))  # 自适应平均池化层

    def forward(self, feat1, feat2, feat3, feat4):
        # feat1: 形状为 (1, 64, 1, 1)
        # feat2: 形状为 (1, 64, 32, 32)
        # feat3: 形状为 (1, 64, 5, 32)
        # feat4: 形状为 (1, 64, 32, 5)

        # 对每个特征图进行自适应平均池化
        pooled_feat1 = self.avg_pool(feat1)  # 结果形状为 (1, 64, 7, 7)
        pooled_feat2 = self.avg_pool(feat2)  # 结果形状为 (1, 64, 7, 7)
        pooled_feat3 = self.avg_pool(feat3)  # 结果形状为 (1, 64, 7, 7)
        pooled_feat4 = self.avg_pool(feat4)  # 结果形状为 (1, 64, 7, 7)

        # 可以在这里对池化后的特征图进行进一步处理，比如拼接
        concatenated_feats = torch.cat((pooled_feat1, pooled_feat2, pooled_feat3, pooled_feat4), dim=1)

        return concatenated_feats

if __name__ == "__main__":
# 示例使用
    model = MyNetwork()
    feat1 = torch.randn(1, 64, 1, 1)
    feat2 = torch.randn(1, 64, 32, 32)
    feat3 = torch.randn(1, 64, 5, 32)
    feat4 = torch.randn(1, 64, 32, 5)
    output = model(feat1, feat2, feat3, feat4)
    print(output.shape)  # 输出形状应该为 (4, 64, 7, 7)