import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


# Helper functions
def to_3d(x):
    b, c, h, w1, w2 = x.shape
    return rearrange(x, 'b c h w1 w2 -> b (h w1 w2) c')


def to_5d(x, h, w1, w2):
    return rearrange(x, 'b (h w1 w2) c -> b c h w1 w2', h=h, w1=w1, w2=w2)


# MatMul Wrapper
class MatMul(nn.Module):
    def __init__(self):
        super(MatMul, self).__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


# LinAngularAttention
class LinAngularAttention(nn.Module):
    def __init__(self, in_channels, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, res_kernel_size=9,
                 sparse_reg=False):
        super().__init__()
        assert in_channels % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = in_channels // num_heads
        self.scale = head_dim ** -0.5
        self.sparse_reg = sparse_reg

        self.qkv = nn.Linear(in_channels, in_channels * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(in_channels, in_channels)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kq_matmul = MatMul()
        self.kqv_matmul = MatMul()
        if self.sparse_reg:
            self.qk_matmul = MatMul()
            self.sv_matmul = MatMul()

        self.dconv = nn.Conv2d(
            in_channels=self.num_heads,
            out_channels=self.num_heads,
            kernel_size=(res_kernel_size, 1),
            padding=(res_kernel_size // 2, 0),
            bias=False,
            groups=self.num_heads,
        )

    def forward(self, x):
        N, L, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(N, L, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)

        if self.sparse_reg:
            attn = self.qk_matmul(q * self.scale, k.transpose(-2, -1)).softmax(dim=-1)
            mask = attn > 0.02
            sparse = mask * attn

        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        dconv_v = self.dconv(v)

        attn = self.kq_matmul(k.transpose(-2, -1), v)

        if self.sparse_reg:
            x = (
                    self.sv_matmul(sparse, v)
                    + 0.5 * v
                    + 1.0 / math.pi * self.kqv_matmul(q, attn)
            )
        else:
            x = 0.5 * v + 1.0 / math.pi * self.kqv_matmul(q, attn)

        x = x / x.norm(dim=-1, keepdim=True)
        x += dconv_v
        x = x.transpose(1, 2).reshape(N, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# Cross-Covariance Attention
class XCA(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        #self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) 
        attn = self.attn_drop(attn.softmax(dim=-1))
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# Cross Attention Fusion Module
class CAFM(nn.Module):
    def __init__(self):
        super(CAFM, self).__init__()
        self.conv1_spatial = nn.Conv2d(2, 1, 3, padding=1)
        self.conv2_spatial = nn.Conv2d(1, 1, 3, padding=1)

        # 修改通道数为64
        self.avg1 = nn.Conv2d(256, 128, 1)
        self.avg2 = nn.Conv2d(256, 128, 1)
        self.max1 = nn.Conv2d(256, 128, 1)
        self.max2 = nn.Conv2d(256, 128, 1)

        self.avg11 = nn.Conv2d(128, 256, 1)
        self.avg22 = nn.Conv2d(128, 256, 1)
        self.max11 = nn.Conv2d(128, 256, 1)
        self.max22 = nn.Conv2d(128, 256, 1)

    def forward(self, f1, f2):
        b, c, h, w = f1.size()
        f1 = f1.reshape(b, c, -1)
        f2 = f2.reshape(b, c, -1)

        def channel_att(x, avg_conv, max_conv, avg_up, max_up):
            avg = F.relu(avg_conv(torch.mean(x, dim=-1, keepdim=True).unsqueeze(-1)))
            max_ = F.relu(max_conv(torch.max(x, dim=-1, keepdim=True)[0].unsqueeze(-1)))
            return avg_up(avg).squeeze(-1) + max_up(max_).squeeze(-1)

        a1 = channel_att(f1, self.avg1, self.max1, self.avg11, self.max11)
        a2 = channel_att(f2, self.avg2, self.max2, self.avg22, self.max22)

        cross = torch.matmul(a1, a2.transpose(1, 2))
        f1_cross = torch.matmul(F.softmax(cross, dim=-1), f1)
        f2_cross = torch.matmul(F.softmax(cross.transpose(1, 2), dim=-1), f2)

        def spatial_att(f):
            f = f.reshape(b, c, h, w)
            avg_out = torch.mean(f, dim=1, keepdim=True)
            max_out = torch.max(f, dim=1, keepdim=True)[0]
            f = F.relu(self.conv1_spatial(torch.cat([avg_out, max_out], dim=1)))
            f = self.conv2_spatial(f).reshape(b, 1, -1)
            return F.softmax(f, dim=-1)

        w1 = spatial_att(f1_cross)
        w2 = spatial_att(f2_cross)

        f1 = (f1 * w1 + f1).transpose(1, 2)
        f2 = (f2 * w2 + f2).transpose(1, 2)
        return f1, f2


# Fusion Module
class LinAngularXCA_CA(nn.Module):
    def __init__(self):
        super(LinAngularXCA_CA, self).__init__()
        # 修改输入通道数为64
        self.la = LinAngularAttention(in_channels=256)
        self.xa = XCA(dim=256)
        self.cafm = CAFM()

    def forward(self, x):
        b, c, h, w1, w2 = x.shape
        # 将5维张量转换为3维张量
        x_3d = to_3d(x)
        la_out_3d = self.la(x_3d)
        xa_out_3d = self.xa(x_3d)

        # 将3维张量转换为4维张量
        la_out_4d = la_out_3d.view(b, h * w1 * w2, c).transpose(1, 2).view(b, c, h * w1, w2)
        xa_out_4d = xa_out_3d.view(b, h * w1 * w2, c).transpose(1, 2).view(b, c, h * w1, w2)

        result1, result2 = self.cafm(la_out_4d, xa_out_4d)
        result_3d = (result1 + result2).transpose(1, 2).reshape(b, h * w1 * w2, c)
        # 将结果转换回5维张量
        result = to_5d(result_3d, h, w1, w2)
        return result
