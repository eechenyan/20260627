import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        self.Wxi = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Whi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Wxf = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Whf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Wxc = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Whc = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Wxo = nn.Conv2d(input_dim, hidden_dim, kernel_size, padding=self.padding)
        self.Who = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=self.padding)

    def forward(self, x, h, c):
        i_t = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        f_t = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        c_t = f_t * c + i_t * torch.tanh(self.Wxc(x) + self.Whc(h))
        o_t = torch.sigmoid(self.Wxo(x) + self.Who(h))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            input_channels = input_dim if i == 0 else hidden_dim
            self.cells.append(ConvLSTMCell(input_channels, hidden_dim, kernel_size))

    def forward(self, x):
        batch_size, _, time_steps, height, width = x.size()
        h = torch.zeros(batch_size, self.cells[0].hidden_dim, height, width, device=x.device)  # 初始化 h
        c = torch.zeros(batch_size, self.cells[0].hidden_dim, height, width, device=x.device)  # 初始化 c

        for t in range(time_steps):  # 遍历时间步
            for layer in range(self.num_layers):
                if layer == 0:
                    h, c = self.cells[layer](x[:, :, t, :, :], h, c)  # 第一层使用输入 x
                else:
                    h, c = self.cells[layer](h, h, c)  # 后续层使用 h

        return h  # 返回最后一个时间步的隐藏状态


# 示例使用
# 定义输入
if __name__ == "__main__":
    input_tensor = torch.randn(8, 960, 27, 1, 1)  # 假设的输入
    model = ConvLSTM(input_dim=960, hidden_dim=64, kernel_size=3, num_layers=2)  # 示例模型
    output = model(input_tensor)  # 前向传播
    print("Output shape:", output.shape)  # 检查输出形状


