import torch
import torch.nn as nn
import torch.nn.functional as F

# Rotary Position Encoding（RoPE）
def apply_rotary_pos_emb(q, k):
    B, H, N, C = q.shape
    half_dim = C // 2
    freq_seq = 1.0 / (10000 ** (torch.arange(0, half_dim, device=q.device).float() / half_dim))
    pos = torch.arange(N, device=q.device).float()

    sinusoid_inp = torch.einsum("n,d->nd", pos, freq_seq)
    sin = sinusoid_inp.sin()[None, None, :, :].repeat(B, H, 1, 1)
    cos = sinusoid_inp.cos()[None, None, :, :].repeat(B, H, 1, 1)

    q1, q2 = q[..., :half_dim], q[..., half_dim:]
    k1, k2 = k[..., :half_dim], k[..., half_dim:]

    q_rot = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1)

    return q_rot, k_rot

# 单头 RoPE Attention（极简）
class UltraLightRoPEAttention(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.drop = nn.Dropout(dropout)
        self.scale = dim ** -0.5

    def forward(self, x):  # x: [B, N, C]
        B, N, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)  # [B, N, C]

        # fake "1 head" for RoPE
        q, k = q.unsqueeze(1), k.unsqueeze(1)  # [B, 1, N, C]
        q, k = apply_rotary_pos_emb(q, k)
        q, k = q.squeeze(1), k.squeeze(1)  # [B, N, C]

        attn = torch.einsum('bqc,bkc->bqk', q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bqk,bkc->bqc', attn, v)
        return self.drop(out)

# 极简 MLP（DWConv 替代）
class UltraLightMLP(nn.Module):
    def __init__(self, dim, drop=0.1):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  # x: [B, N, C]
        x = x.transpose(1, 2)  # -> [B, C, N]
        x = self.dwconv(x)
        x = self.drop(self.act(x))
        return x.transpose(1, 2)  # -> [B, N, C]

# 极简 TransformerBlock
class TransformerBlock(nn.Module):
    def __init__(self, dim, depth=1, drop=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "norm1": nn.LayerNorm(dim),
                "attn": UltraLightRoPEAttention(dim, drop),
                "norm2": nn.LayerNorm(dim),
                "mlp": UltraLightMLP(dim, drop)
            }) for _ in range(depth)
        ])

    def forward(self, x):  # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B, -1, C)  # -> [B, N, C]

        for layer in self.layers:
            x = x + layer["attn"](layer["norm1"](x))
            x = x + layer["mlp"](layer["norm2"](x))

        x = x.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        return x
if __name__ == "__main__":
    # 计算参数量
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # 初始化模型
    dim = 128
    depth = 2
    drop = 0.1
    model = UltraLightTransformerBlock(dim, depth, drop)

    # 计算参数量
    param_count = count_parameters(model)
    print(f"模型参数量: {param_count}")
