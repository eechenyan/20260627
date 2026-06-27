from CBAM1 import *
from XCA import *
from ResBlock1 import *
from ResBlock2 import *
from GCNet1 import *
from MyAvgnet import MyNetwork
import time  # 导入时间模块


# //自己网络就是定义结构，之后forward描述数据处理过程
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # part1-------------------------------------------------------------------------------------------------------
        # 这部分是用于特征分离
        # 这里的输出尺寸要一致。
        # AFE=64×27×1×1  # SFE=64×9×32×32 EFE_H=64×27×5×32 EHE_V=64×27×32×5
        # -------------------------------------------------------------这下面是SAI的解耦-----------------------------------------------------------------------
        self.AFE_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 32, 32),
                                   stride=(3, 32, 32))  # 具体说明请参照笔记con3d.md,out_channels=64说明有64个卷积核
        self.AFE_conv1_bn = nn.BatchNorm3d(64)  # 归一化
        self.SFE_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(9, 1, 1), stride=(1, 1, 1),
                                   dilation=(9, 1, 1))  # dilation=(7,1,1)重点解释 9意思是卷积核膨胀7倍，中间为0，即1和8才有数字，其他都是0。1代表不变
        self.SFE_conv1_bn = nn.BatchNorm3d(64)
        # -------------------------------------------------------------这下面是EPI的解耦-----------------------------------------------------------------------
        self.EFE_H_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 16, 1), stride=(3, 4, 1))
        self.EFE_H_conv1_bn = nn.BatchNorm3d(64)
        self.EFE_V_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3, 1, 16), stride=(3, 1, 4))
        self.EFE_V_conv1_bn = nn.BatchNorm3d(64)
        # 此处对于每一个patch分离了四个特征出来
        # -------------------------------------------------------------------------------------------------------
        # part2--------------------------------------------------------------------------------------------------------
        # 这部分是进行CBAM特征增强
        # 实例化每一个CBAM 对于每一个特征进行CBAM处理 它处理的是五维的张量请注意 输出的尺寸大小不变
        self.CBAM_AFE = LinAngularXCA_CA()
        self.CBAM_SFE = LinAngularXCA_CA()
        self.CBAM_EFE_H = LinAngularXCA_CA()
        self.CBAM_EFE_V = LinAngularXCA_CA()
        # -------------------------------------------------------------------------------------------------------------------
        # part3-------------------------------------------------------------------------------------------------------------------------
        # 这部分是三层残差网络
        # 第一层（堆叠残差块）
        self.residual_block1_1 = ResidualBlock1(in_channels=64)  # 第一个残差块
        self.residual_block1_2 = ResidualBlock2(in_channels=128)  # 第二个残差块
        # AFE输出尺寸128 × 27 × 1 × 1 SFE=128 × 9 × 32 × 32 EFE_H=128 × 27 × 5 × 32 EHE_V=128×27×32×5
        # ----------------------------------------------------------------------------------------------------------------------
        # part4------------------------------------------------------------------------------------------------------------------
        # 这部分融合特征和进行金字塔拼接
        self.GCNet_AFE = GCNet(in_channels=192, output_channels=1)  # 输入特征数为192 #hidden_dim可以是64 num_layers可以是2
        self.GCNet_SFE = GCNet(in_channels=192, output_channels=1)  # 输入特征数为192# 应为 (1, hidden_dim, 32, 32)
        self.GCNet_EFE_H = GCNet(in_channels=192, output_channels=1)  # 输入特征数为192# 应为 (1, hidden_dim, 5, 32)
        self.GCNet_EHE_V = GCNet(in_channels=192, output_channels=1)  # 输入特征数为192  # 应为 (1, hidden_dim, 32, 5)
        # 金字塔池化拼接
        self.avg_pool = MyNetwork()  # 这个层可以把上面四层的的输出1*27*1*1 1*27*32*32 1*27*5*32 1*27*32*5 一起处理统一为输出1 27 5 5 并完成拼接
        # ------------------------------------------------------------------------------------------------------------------
        # part5----------------------------------------------------------------------------------------------------------------------
        # 这部分是预测分数，记得要在forward展平
        self.FC1 = nn.Linear(4 * 27 * 5 * 5, 512)  # 第一层，输入维度为 4*24*5*5，输出维度为 512
        self.FC2 = nn.Linear(512, 128)  # 第二层，输入维度为 512，输出维度为 128
        self.FC3 = nn.Linear(128, 1)  # 第三层，输入维度为 128，输出维度为 1
        # ------------------------------------------------------------------------------------------------------------------
        # part6-------------------------------------------------------------------------------------------------------------------------------
        # 激活函数
        self.relu = nn.LeakyReLU()
        # -----------------------------------------------------------------------------------------------------------------------------

    def forward(self, x):
        # 1.四幻神阶段
        AFE_out = self.AFE_conv1_bn(self.AFE_conv1(x))
        AFE_out = self.relu(AFE_out)

        SFE_out = self.SFE_conv1_bn(self.SFE_conv1(x))
        SFE_out = self.relu(SFE_out)

        EFE_H_out = self.EFE_H_conv1_bn(self.EFE_H_conv1(x))
        EFE_H_out = self.relu(EFE_H_out)

        EFE_V_out = self.EFE_V_conv1_bn(self.EFE_V_conv1(x))
        EFE_V_out = self.relu(EFE_V_out)

        # 2.进行CBAM特征增强阶段和获取weight阶段
        AFE_CBAM_out = self.CBAM_AFE(AFE_out)
        SFE_CBAM_out = self.CBAM_SFE(SFE_out)
        EFE_H_CBAM_out = self.CBAM_EFE_H(EFE_H_out)
        EFE_V_CBAM_out = self.CBAM_EFE_V(EFE_V_out)

        # 3.进行残差网络阶段
        AFE_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(AFE_CBAM_out, AFE_out))
        SFE_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(SFE_CBAM_out, SFE_out))
        EFE_H_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(EFE_H_CBAM_out, EFE_H_out))
        EFE_V_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(EFE_V_CBAM_out, EFE_V_out))

        # 4.拼接阶段
        AFE_CBAM_out_Rcat = torch.cat((AFE_CBAM_out, AFE_CBAM_out_R2), dim=1)
        SFE_CBAM_out_Rcat = torch.cat((SFE_CBAM_out, SFE_CBAM_out_R2), dim=1)
        EFE_H_CBAM_out_Rcat = torch.cat((EFE_H_CBAM_out, EFE_H_CBAM_out_R2), dim=1)
        EFE_V_CBAM_out_Rcat = torch.cat((EFE_V_CBAM_out, EFE_V_CBAM_out_R2), dim=1)

        # AFE_CBAM_out: [batch_size, 64, 27, 1, 1]
        # AFE_CBAM_out_R2: [batch_size, 128, 27, 1, 1]

        # print("AFE_CBAM_out_Rcat shape:", AFE_CBAM_out_Rcat.shape)  # 应为 [batch_size, 192, 27, 1, 1]
        # 5.ConvLstm阶段
        AFE_CBAM_GCNet_out = self.GCNet_AFE(AFE_CBAM_out_Rcat)
        SFE_CBAM_GCNet_out = self.GCNet_SFE(SFE_CBAM_out_Rcat)
        EFE_H_CBAM_GCNet_out = self.GCNet_EFE_H(EFE_H_CBAM_out_Rcat)
        EFE_V_CBAM_GCNet_out = self.GCNet_EHE_V(EFE_V_CBAM_out_Rcat)

        # 6.金字塔池化拼接阶段
        FC = self.avg_pool(AFE_CBAM_GCNet_out, SFE_CBAM_GCNet_out, EFE_H_CBAM_GCNet_out, EFE_V_CBAM_GCNet_out)

        # 7.预测分数阶段
        # 展平张量为 (8, 27 * 5 * 5)
        FC = FC.view(FC.size(0), -1)  # 这将变为 (8, 27 * 5 * 5)

        # 输入到全连接层
        FC1 = self.FC1(FC)  # 通过全连接层
        FC1 = self.relu(FC1)

        FC2 = self.FC2(FC1)

        FC2 = self.relu(FC2)
        Score = self.FC3(FC2)
        # ----------------------------------------------------------------
        # 多卡训练的代码 调整形状
        # 确保 Score 的形状是 (batch_size, 1)
        # Score = Score.unsqueeze(1)  # 将标量转换为 (batch_size, 1)
        # -----------------------------------------------------------
        return Score
        # 建议后续训练的时候打印出weight和FC2的分数 保存在excel中 以便后续调整
        # 源码是在train里面进行加权求和的


if __name__ == "__main__":
    # 创建模型实例
    model = Network()

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    total_params_m = total_params / (1024 * 1024)  # 转换为 MB

    print(f"Total parameters: {total_params:,} ({total_params_m:.2f} MB)")

    # 创建随机输入数据，形状为 (8, 1, 81, 32, 32)
    data = torch.randn(8, 1, 81, 32, 32)

    # 开始计时
    start_time = time.time()

    # 进行前向传播
    score = model(data)

    # 结束计时
    end_time = time.time()

    # 打印输出和形状

    print("Time taken for forward pass: {:.4f} seconds".format(end_time - start_time))
