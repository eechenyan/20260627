import torch
import torch.nn as nn

class GCNet(nn.Module):
    def __init__(self, in_channels, output_channels):
        super(GCNet, self).__init__()
        # 使用 1x1 卷积将 192 个通道融合为 1 个通道
        self.channel_fusion = nn.Conv3d(
            in_channels=in_channels,
            out_channels=output_channels,
            kernel_size=1
        )

    def forward(self, x):
        # 输入 x shape: (192, 27, 1, 1)
        # x = x.unsqueeze(0)  # 添加 batch 维度，变为 (1, 192, 27, 1, 1)

        # 通道融合
        x = self.channel_fusion(x)  # 结果 shape: (1, 1, 27, 1, 1)

        return x  # 去掉 batch 维度，变回 (1, 27, 1, 1)
        
if __name__ == "__main__":

    
    # 创建模型
    model = GCNet(in_channels=192, output_channels=1)
    
    # 创建一个示例输入张量，形状为 (N, in_channels, H, W)
    input_tensor = torch.rand(192, 27, 1, 1)  # 1个样本，192个通道，32x32的特征图
    
    # 进行前向传播
    output = model(input_tensor)
    
    
    
    # 打印输出形状
    print("Output shape:", output.shape)  # 期望形状: (1, output_channels, 32, 32)




