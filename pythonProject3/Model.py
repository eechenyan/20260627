import torch
import torch.nn as nn


# //自己网络就是定义结构，之后forward描述数据处理过程
class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        # 输入是1*25*32*32
        self.h_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5,1,1), stride=(5,1,1))
        # 输出64×5×32×32
        #stride=(5,1,1) 5意味着时间维度上移动5，1意味着宽度和高度维度上面移动1，每次等高度和宽度层面卷积完成之后在进行时间移动5位。和你想的没有问题。注意！！这个stride和2d卷积的三个参数意义完全不一样
        #out_channels=64说明有64个卷积核
        #具体说明请参照笔记con3d.md
        self.h_conv1_bn = nn.BatchNorm3d(64) #归一化
        self.v_conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=(5,1,1), stride=(1,1,1), dilation=(5,1,1))
        #dilation=(5,1,1)重点解释 5意思是卷积核膨胀5倍，中间为0，即1和6才有数字，其他都是0，1代表不变
        self.v_conv1_bn = nn.BatchNorm3d(64)
        # 64×21×32×32

        ### hor
        self.h_conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        #kernel_size=(1, 3, 3)其实意思是二维卷积
        #这里输出应该是64*分辨率*分辨率 *64 最后64代表有64个 ？？存疑 这是3d的理解 但是是否要按照2d去解释？
        self.h_conv2_bn = nn.BatchNorm3d(64)
        self.h_conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.h_conv3_bn = nn.BatchNorm3d(128)
        self.h_conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.h_conv4_bn = nn.BatchNorm3d(256)
        self.h_conv5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.h_conv5_bn = nn.BatchNorm3d(256)

        ### ver
        self.v_conv2 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.v_conv2_bn = nn.BatchNorm3d(64)
        self.v_conv3 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.v_conv3_bn = nn.BatchNorm3d(128)
        self.v_conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.v_conv4_bn = nn.BatchNorm3d(256)
        self.v_conv5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.v_conv5_bn = nn.BatchNorm3d(256)

        #输入64个通道，输出256个通道，特征提取更加丰富了

        self.h_GRU = MyGRU(256, 256, batch_first=True)
        self.v_GRU = MyGRU(256, 256, batch_first=True)

        ### regression
        self.linear_1 = nn.Linear(512, 128)
        self.linear_2 = nn.Linear(128, 1)

        self.relu = nn.LeakyReLU()
        self.avg_pooling = nn.AvgPool3d(kernel_size=(1, 16, 16), stride=1, padding=0)
        self.flat = nn.Flatten()

    def forward(self, x):

        h_x = self.relu(self.h_conv1_bn(self.h_conv1(x)))
        #三个过程：x经过卷积 然后归一化 最后激活
        v_x = self.relu(self.v_conv1_bn(self.v_conv1(x)))

        h_x = self.relu(self.h_conv2_bn(self.h_conv2(h_x)))
        h_x = self.relu(self.h_conv3_bn(self.h_conv3(h_x)))
        h_x = self.relu(self.h_conv4_bn(self.h_conv4(h_x)))
        h_x = self.relu(self.h_conv5_bn(self.h_conv5(h_x)))
        h_x = self.avg_pooling(h_x)
        h_x = self.h_GRU(h_x)

        v_x = self.relu(self.v_conv2_bn(self.v_conv2(v_x)))
        v_x = self.relu(self.v_conv3_bn(self.v_conv3(v_x)))
        v_x = self.relu(self.v_conv4_bn(self.v_conv4(v_x)))
        v_x = self.relu(self.v_conv5_bn(self.v_conv5(v_x)))
        v_x = self.avg_pooling(v_x)
        v_x = self.v_GRU(v_x)

        out = torch.cat([h_x,v_x],dim=1)
        out = self.linear_1(out)
        out = self.relu(out)
        out = self.linear_2(out)

        return out

class MyGRU(nn.Module):
    # input: n c T 1 1
    # output: n c

    # def __init__(self, input_size, hidden_size, device="cuda:0", batch_first=True):
    def __init__(self, input_size, hidden_size, device="cpu", batch_first=True):
        super(MyGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=batch_first)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.hidden_size = hidden_size
        self.device = device

    def forward(self, x):  # n c T 1 1
        t = torch.squeeze(x, dim=3)
        t = torch.squeeze(t, dim=3)
        t = t.permute([0, 2, 1])
        r, h1 = self.rnn(t, self._get_initial_state(t.size(0), self.device))
        r = r.permute([0, 2, 1])
        f = self.pool(r).squeeze(2)
        return f

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0

