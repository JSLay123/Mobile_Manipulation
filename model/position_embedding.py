import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=910):
        super(PositionalEmbedding, self).__init__()
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arrange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)
        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1), :]
        return x

class PositionalEmbedding2D(nn.Module):
    def __init__(self, d_model=512, height=20, width=15):
        """
        参数:
            d_model: token 的维度 (如 512)
            height, width: 2D 网格的高度和宽度
        """
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("d_model must be even for 2D positional encoding.")
            
        self.d_model = d_model
        self.height = height
        self.width = width
        
        # (height, width, d_model)
        pe = torch.zeros(height, width, d_model)
        d_half = d_model // 2
        
        # 为高度创建位置编码
        pos_h = torch.arange(height).unsqueeze(1).float()  # (height, 1)
        div_term_h = torch.exp(torch.arange(0, d_half, 2).float() * 
                              -(math.log(10000.0) / d_half))  # (d_half/2,)
        
        # 为宽度创建位置编码
        pos_w = torch.arange(width).unsqueeze(1).float()  # (width, 1)
        div_term_w = torch.exp(torch.arange(0, d_half, 2).float() * 
                              -(math.log(10000.0) / d_half))  # (d_half/2,)
        
        pe[:, :, 0:d_half:2] = torch.sin(pos_h * div_term_h).unsqueeze(1)
        pe[:, :, 1:d_half:2] = torch.cos(pos_h * div_term_h).unsqueeze(1)
        
        pe[:, :, d_half::2] = torch.sin(pos_w * div_term_w).unsqueeze(0)
        pe[:, :, d_half+1::2] = torch.cos(pos_w * div_term_w).unsqueeze(0)
        
        # 展平为 (height*width, d_model) 即 (300, 512)
        pe = pe.reshape(-1, d_model)
        self.register_buffer('pe', pe)  # 不参与梯度更新

    def forward(self, x, num_views=1):
        """
        输入 x shape: [B, L, D]
        
        参数:
            x: 输入token，形状为 [batch_size, sequence_length, feature_dim]
            num_views: 视角数量（默认为1，多视角时传入实际值）
        返回:
            添加了位置编码的token
        """
        B, L, D = x.shape
        
        # 计算每个视角的token数
        tokens_per_view = self.height * self.width
        
        # 验证序列长度是否匹配
        if L != tokens_per_view * num_views:
            raise ValueError(f"序列长度 L={L} 不匹配预期值 {tokens_per_view} * {num_views} = {tokens_per_view * num_views}")
        
        # 将输入重塑为 [B, num_views, tokens_per_view, D]
        x_reshaped = x.view(B, num_views, tokens_per_view, D)
        
        # 添加位置编码
        # self.pe 形状为 [tokens_per_view, D]
        x_with_pe = x_reshaped + self.pe.unsqueeze(0).unsqueeze(0)  # 广播到 [B, num_views, tokens_per_view, D]
        
        # 还原回原始形状
        return x_with_pe.view(B, L, D)
