import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.jit import Final
from timm.models.vision_transformer import Attention, Mlp, RmsNorm, use_fused_attn

class TimeEmbedding(nn.Module):
    """
    input:[B,]
    output:[B, hidden_size=512]
    """
    def __init__(self, hidden_size=512, embedding_size=1024):
        super(TimeEmbedding, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )
        self.embedding_size = embedding_size
        # self.dtype = dtype

    def time_embedding(self, t, dim, max_period=10000):
        # 生成时间步长的正弦和余弦位置编码
        half_dim = dim // 2
        frequencies = torch.exp(
            -math.log(max_period) * torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        # print(frequencies.shape)
        # print(t.shape)
        args = t[:, None].float() * frequencies[None, :]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding
    
    def forward(self, t):
        # 生成时间步长的嵌入
        time_emb = self.time_embedding(t, self.embedding_size)
        # 通过MLP将时间嵌入映射到隐藏空间
        return self.mlp(time_emb)
    

class CrossAttention(nn.Module):
    """
    A cross-attention layer with flash attention.
    """
    fused_attn = Final[bool]
    def __init__(
                self,
                dim: int,
                num_heads: int = 8,
                qkv_bias: bool = False,
                qk_norm: bool = False,
                attn_drop: float = 0,
                proj_drop: float = 0,
                norm_layer: nn.Module = nn.LayerNorm
                ) -> None:
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x : torch.Tensor, c : torch.Tensor, 
                mask : torch.Tensor | None = None) -> torch.Tensor:
        B, N, C = x.shape
        _, L, _ = c.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(c).reshape(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        # print([DEBUG]:", q.shape, k.shape, v.shape)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, L)
            mask = mask.expand(-1, -1, N, -1)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        if mask is not None:
            attn = attn.masked_fill(mask.logical_not(), float('-inf'))
        attn = attn.softmax(dim=-1)
        if self.attn_drop.p > 0:
            attn = self.attn_drop(attn)
        x = attn @ v

        x = x.permute(0, 2, 1, 3).reshape(B, N, C)
        x = self.proj(x)
        if self.proj_drop.p > 0:
            x = self.proj_drop(x)
        return x



class DiT_Block(nn.Module):
    """
    DiT Block with cross_attn conditioning.
    """
    def __init__(self, hidden_size, num_heads):
        super(DiT_Block, self).__init__()
        self.norm1 = RmsNorm(hidden_size, 1e-6)
        # self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = Attention(
            dim = hidden_size, num_heads=num_heads, 
            qkv_bias=True, qk_norm=True, 
            norm_layer=RmsNorm,
        )
        self.cross_attn = CrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True, qk_norm=True,
            norm_layer=RmsNorm,
        )

        self.norm2 = RmsNorm(hidden_size, eps=1e-6)
        # self.norm2 = nn.LayerNorm(hidden_size)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn = Mlp(in_features=hidden_size, 
            hidden_features=hidden_size, 
            act_layer=approx_gelu, drop=0)
        self.norm3 = RmsNorm(hidden_size, eps=1e-6)

    def forward(self, x, c, mask=None):
        origin_x = x
        x = self.norm1(x)
        x = self.attn(x)
        x = x + origin_x

        origin_x = x
        x = self.norm2(x)
        print(x.shape)
        print(c.shape)
        x = self.cross_attn(x, c, mask)
        print(x.shape)
        x = x + origin_x

        origin_x = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + origin_x

        return x


class FinalLayer(nn.Module):
    """
    The final layer after DiT.
    """
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.ffn_final = Mlp(in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_dim, 
            act_layer=approx_gelu, drop=0)

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x

def test_dit_architecture():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 2
    hidden_size = 512
    action_seq_len = 10  # 假设动作序列长度
    vision_token_len = 900 # 3视角拼接后的长度
    out_dim = 1

    print(f"--- 开始架构 Shape 验证 (Device: {device}) ---")

    # 1. 测试 TimeEmbedding
    time_net = TimeEmbedding(hidden_size=hidden_size).to(device)
    t = torch.randn(2).to(device)
    t_emb = time_net(t)
    print(f"Time Embedding Output: {t_emb.shape}") # 预期: [2, 512]
    assert t_emb.shape == (batch_size, hidden_size)

    # 2. 准备 DiT_Block 的输入
    # x 通常是当前的 noisy action 序列
    x = torch.randn(batch_size, action_seq_len, hidden_size).to(device)
    # c 是外部条件 (如视觉特征)
    c = torch.randn(batch_size, vision_token_len, hidden_size).to(device)

    # print(x.shape, c.shape)
    
    # 3. 测试 DiT_Block
    # 注意：你的代码中 CrossAttention 初始化参数名是 hidden_size，但类定义里是 dim
    # 这里需要确保参数名对应。根据你提供的类定义，初始化应传 dim
    block = DiT_Block(hidden_size=hidden_size, num_heads=8).to(device)
    
    # 执行前向传播
    block_output = block(x, c)
    print(f"DiT Block Output: {block_output.shape}") # 预期: [2, 10, 512]
    assert block_output.shape == (batch_size, action_seq_len, hidden_size)

    # 4. 测试 FinalLayer
    final_layer = FinalLayer(hidden_size=hidden_size, out_dim=out_dim).to(device)
    final_out = final_layer(block_output)
    print(f"Final Layer Output: {final_out.shape}") # 预期: [2, 10, 7]
    assert final_out.shape == (batch_size, action_seq_len, out_dim)

    print("--- 所有模块 Shape 验证通过！ ---")

if __name__ == "__main__":
    # time_emb = TimeEmbedding()
    # t = torch.tensor([0, 1, 2, 3])
    # print(time_emb(t).shape)
    test_dit_architecture()
