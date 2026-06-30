import torch
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((5, 5))  # 自适应平均池化层

    def forward(self, feat1, feat2, feat3, feat4):
        # feat1: 形状为 (1, 27, 1, 1)
        # feat2: 形状为 (1, 27, 32, 32)
        # feat3: 形状为 (1, 27, 5, 32)
        # feat4: 形状为 (1, 27, 32, 5)

        # 对 feat1 进行处理
        pooled_feat1 = self.avg_pool(feat1.squeeze(1))  # 结果形状为 (8, 27, 5, 5)

        # 对 feat2 进行自适应平均池化
        pooled_feat2 = self.avg_pool(feat2.squeeze(1))  # 结果形状为 (8, 27, 5, 5)
        pooled_feat2 = pooled_feat2.repeat(1, 3, 1, 1)  # 重复通道以达到 27

        # 对 feat3 进行自适应平均池化
        pooled_feat3 = self.avg_pool(feat3.squeeze(1))  # 结果形状为 (8, 27, 5, 5)

        # 对 feat4 进行自适应平均池化
        pooled_feat4 = self.avg_pool(feat4.squeeze(1))  # 结果形状为 (8, 27, 5, 5)
        

       
        
        # 将所有池化后的特征图在通道维度上拼接
        concatenated_feats = torch.cat((pooled_feat1, pooled_feat2, pooled_feat3, pooled_feat4), dim=1)

        return concatenated_feats

# 示例用法
if __name__ == "__main__":
    model = MyNetwork()

    # 创建示例输入
    feat1 = torch.rand(1, 27, 1, 1)          
    feat2 = torch.rand(1, 27, 32, 32)        
    feat3 = torch.rand(1, 27, 5, 32)         
    feat4 = torch.rand(1, 27, 32, 5)         

    # 进行前向传播
    output = model(feat1, feat2, feat3, feat4)
    print("Output shape:", output.shape)  # 期望形状: (8, 27*4, 5, 5)