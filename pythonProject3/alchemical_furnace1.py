from CBAM1 import *
from ResBlock1 import *
from ResBlock2 import *
from ConvLstm1 import *
from MyAvgnet import MyNetwork


# //自己网络就是定义结构，之后forward描述数据处理过程
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # part1-------------------------------------------------------------------------------------------------------
        # 这部分是用于特征分离
        # 这里的输出尺寸要一致。
        # AFE=64×27×1×1  # SFE=64×9×32×32 EFE_H=64×27×5×32 EHE_V=64×27×32×5
        self.AFE_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,32,32), stride=(3,32,32))#具体说明请参照笔记con3d.md,out_channels=64说明有64个卷积核
        self.AFE_conv1_bn = nn.BatchNorm3d(64) #归一化
        self.SFE_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(9,1,1), stride=(1,1,1), dilation=(9,1,1))#dilation=(9,1,1)重点解释 9意思是卷积核膨胀9倍，中间为0，即1和10才有数字，其他都是0。1代表不变
        self.SFE_conv1_bn = nn.BatchNorm3d(64)
        self.EFE_H_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,16,1), stride=(3,4,1))
        self.EFE_H_conv1_bn = nn.BatchNorm3d(64)
        self.EFE_V_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(3,1,16), stride=(3,1,4))
        self.EFE_V_conv1_bn = nn.BatchNorm3d(64)
        # 此处对于每一个patch分离了四个特征出来
        # -------------------------------------------------------------------------------------------------------

        # part2--------------------------------------------------------------------------------------------------------
        # 这部分是进行CBAM特征增强
        # 实例化每一个CBAM 对于每一个特征进行CBAM处理 它处理的是五维的张量请注意 输出的尺寸大小不变
        self.CBAM_AFE = CBAM(in_channels=64)
        self.CBAM_SFE = CBAM(in_channels=64)
        self.CBAM_EFE_H = CBAM(in_channels=64)
        self.CBAM_EFE_V = CBAM(in_channels=64)

        #实例化 计算某一PBVS权重的层 之后在forward方法中使用它的方法就行了
        self.weight = CWT()
        #-------------------------------------------------------------------------------------------------------------------

        # part3-------------------------------------------------------------------------------------------------------------------------
        # 这部分是三层残差网络
        # 第一层（堆叠残差块）
        self.residual_block1_1 = ResidualBlock1(in_channels=64)  #第一个残差块
        self.residual_block1_2 = ResidualBlock2(in_channels=128) #第二个残差块
        # AFE输出尺寸128 × 27 × 1 × 1 SFE=128 × 9 × 32 × 32 EFE_H=128 × 27 × 5 × 32 EHE_V=128×27×32×5

        #第二层
        self.residual_block2_1 = ResidualBlock1(in_channels=128)  # 第二个残差层
        self.residual_block2_2 = ResidualBlock2(in_channels=256)
        # AFE=出尺寸256 × 27 × 1 × 1 SFE=256 × 9 × 32 × 32 EFE_H=256 × 27 × 5 × 32 EHE_V=256×27×32×5

        #第三层
        self.residual_block3_1 = ResidualBlock1(in_channels=256)  # 第三个残差层
        self.residual_block3_2 = ResidualBlock2(in_channels=512)
        # AFE输出尺寸：512 × 27 × 1 × 1 SFE=512 × 9 × 32 × 32 EFE_H=512 × 27 × 5 × 32 EHE_V=512×27×32×5
        # ----------------------------------------------------------------------------------------------------------------------

        # part4------------------------------------------------------------------------------------------------------------------
        # 这部分融合特征和进行金字塔拼接
        self.convlstm_AFE = ConvLSTM(input_dim=960, hidden_dim=64, kernel_size=1, num_layers=1) #hidden_dim可以是64 num_layers可以是2
        self.convlstm_SFE = ConvLSTM(input_dim=960, hidden_dim=64, kernel_size=3, num_layers=1) # 应为 (1, hidden_dim, 32, 32)
        self.convlstm_EFE_H = ConvLSTM(input_dim=960, hidden_dim=64, kernel_size=3, num_layers=1) # 应为 (1, hidden_dim, 5, 32)
        self.convlstm_EHE_V = ConvLSTM(input_dim=960, hidden_dim=64, kernel_size=3, num_layers=1)  # 应为 (1, hidden_dim, 32, 5)
        # 金字塔池化拼接
        self.avg_pool = MyNetwork() #这个层可以把上面四层的的输出1*64*1*1 1*64*32*32 1*64*5*32 1*64*32*5 一起处理统一为输出1 64 7 7 并完成拼接
        #------------------------------------------------------------------------------------------------------------------

        # part5----------------------------------------------------------------------------------------------------------------------
        # 这部分是预测分数，记得要在forward展平
        self.FC1 = nn.Linear( 4* 64 * 7 * 7, 128)  # 第一层，输入维度为 64*7*7，输出维度为 128
        self.FC2 = nn.Linear(128, 1)  # 第二层，输入维度为 128，输出维度为 1
        #------------------------------------------------------------------------------------------------------------------

        # part6-------------------------------------------------------------------------------------------------------------------------------
        # 激活函数
        self.relu = nn.LeakyReLU()
        #-----------------------------------------------------------------------------------------------------------------------------

    def forward(self,x):
        #1.四幻神阶段
        AFE_out = self.AFE_conv1_bn(self.AFE_conv1(x))
        AFE_out= self.relu(AFE_out)

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
        weight = self.weight(AFE_CBAM_out,SFE_CBAM_out,EFE_H_CBAM_out,EFE_V_CBAM_out)

        # 3.进行残差网络阶段
        AFE_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(AFE_CBAM_out,AFE_out))
        AFE_out = torch.cat((AFE_out, AFE_out), dim=1)  # 在通道维度上拼接
        AFE_CBAM_out_R3 = self.residual_block2_2(self.residual_block2_1(AFE_CBAM_out_R2,AFE_out))
        AFE_out = torch.cat((AFE_out, AFE_out), dim=1)  # 在通道维度上拼接
        AFE_CBAM_out_R4 = self.residual_block3_2(self.residual_block3_1(AFE_CBAM_out_R3, AFE_out))

        SFE_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(SFE_CBAM_out,SFE_out))
        SFE_out = torch.cat((SFE_out, SFE_out), dim=1)  # 在通道维度上拼接
        SFE_CBAM_out_R3 = self.residual_block2_2(self.residual_block2_1(SFE_CBAM_out_R2,SFE_out))
        SFE_out = torch.cat((SFE_out, SFE_out), dim=1)
        SFE_CBAM_out_R4 = self.residual_block3_2(self.residual_block3_1(SFE_CBAM_out_R3, SFE_out))

        EFE_H_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(EFE_H_CBAM_out,EFE_H_out))
        EFE_H_out = torch.cat((EFE_H_out, EFE_H_out), dim=1)
        EFE_H_CBAM_out_R3 = self.residual_block2_2(self.residual_block2_1(EFE_H_CBAM_out_R2,EFE_H_out))
        EFE_H_out = torch.cat((EFE_H_out, EFE_H_out), dim=1)
        EFE_H_CBAM_out_R4 = self.residual_block3_2(self.residual_block3_1(EFE_H_CBAM_out_R3, EFE_H_out))

        EFE_V_CBAM_out_R2 = self.residual_block1_2(self.residual_block1_1(EFE_V_CBAM_out,EFE_V_out))
        EFE_V_out = torch.cat((EFE_V_out, EFE_V_out), dim=1)
        EFE_V_CBAM_out_R3 = self.residual_block2_2(self.residual_block2_1(EFE_V_CBAM_out_R2,EFE_V_out))
        EFE_V_out = torch.cat((EFE_V_out, EFE_V_out), dim=1)
        EFE_V_CBAM_out_R4 = self.residual_block3_2(self.residual_block3_1(EFE_V_CBAM_out_R3, EFE_V_out))

        #4.拼接阶段
        AFE_CBAM_out_Rcat= torch.cat((AFE_CBAM_out,AFE_CBAM_out_R2,AFE_CBAM_out_R3,AFE_CBAM_out_R4),dim=1)
        SFE_CBAM_out_Rcat = torch.cat((SFE_CBAM_out, SFE_CBAM_out_R2, SFE_CBAM_out_R3, SFE_CBAM_out_R4), dim=1)
        EFE_H_CBAM_out_Rcat = torch.cat((EFE_H_CBAM_out, EFE_H_CBAM_out_R2, EFE_H_CBAM_out_R3, EFE_H_CBAM_out_R4), dim=1)
        EFE_V_CBAM_out_Rcat = torch.cat((EFE_V_CBAM_out, EFE_V_CBAM_out_R2, EFE_V_CBAM_out_R3, EFE_V_CBAM_out_R4), dim=1)

        # AFE_CBAM_out: [batch_size, 64, 27, 1, 1]
        # AFE_CBAM_out_R2: [batch_size, 128, 27, 1, 1]
        # AFE_CBAM_out_R3: [batch_size, 256, 27, 1, 1]
        # AFE_CBAM_out_R4: [batch_size, 512, 27, 1, 1]
        # AFE_CBAM_out_Rcat: [batch_size, 960, 27, 1, 1]
        # print("AFE_CBAM_out_Rcat shape:", AFE_CBAM_out_Rcat.shape)  # 应为 [batch_size, 960, 27, 1, 1]
        #5.ConvLstm阶段
        AFE_CBAM_CLstm_out = self.convlstm_AFE(AFE_CBAM_out_Rcat)
        SFE_CBAM_CLstm_out = self.convlstm_SFE(SFE_CBAM_out_Rcat)
        EFE_H_CBAM_CLstm_out = self.convlstm_EFE_H(EFE_H_CBAM_out_Rcat)
        EFE_V_CBAM_CLstm_out = self.convlstm_EHE_V(EFE_V_CBAM_out_Rcat)

        #6.金字塔池化拼接阶段
        FC = self.avg_pool(AFE_CBAM_CLstm_out,SFE_CBAM_CLstm_out,EFE_H_CBAM_CLstm_out,EFE_V_CBAM_CLstm_out)

        #7.预测分数阶段
        # 展平张量为 (8, 64 * 7 * 7)
        FC = FC.view(FC.size(0), -1) # 这将变为 (8, 64 * 7 * 7)
        # 输入到全连接层
        FC1 = self.FC1(FC)  # 通过全连接层
        FC1 = self.relu(FC1)
        Score = self.FC2(FC1)
#----------------------------------------------------------------
        # 多卡训练的代码 调整形状
        # 确保 Score 的形状是 (batch_size, 1)
        #Score = Score.unsqueeze(1)  # 将标量转换为 (batch_size, 1)
#-----------------------------------------------------------
        return Score,weight
        #建议后续训练的时候打印出weight和FC2的分数 保存在excel中 以便后续调整
        #源码是在train里面进行加权求和的

