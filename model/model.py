import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, cast, List

# 保持原来的 SinusoidalTimeEmbedding 不变
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, hidden_dim: int):
        super(SinusoidalTimeEmbedding, self).__init__()
        self.hidden_dim = hidden_dim

    def forward(self, t):
        if t.dim() == 1: t = t.unsqueeze(-1)
        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.hidden_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

class ConditionalEditFlowsTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int,
        num_layers: int,
        VDJ_embedding_dim: int,
        num_heads=8,
        max_seq_len=512,
        pad_token_id=129,
    ):
        super(ConditionalEditFlowsTransformer, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        # --- Embedding 层 ---
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        # position embedding 需要额外容纳3个VDJ embeddings位置
        self.pos_embedding = nn.Embedding(max_seq_len + 3, hidden_dim)
        ## VDJ embeddings
        self.VDJ_layers = nn.Sequential(
            nn.Linear(VDJ_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # 先做一次归一化
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(0.1) # 增加随机性，防止模型过度依赖这个强信号
        )
        ## time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim=hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        ## time projection
        self.time_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, num_layers * 2 * hidden_dim) # 每层两个参数
        )

        # --- 主编码器 (Encoder) ---
        ## 使用 Self-Attention，VDJ embeddings 作为 prefix
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1, 
                activation='gelu',
                batch_first=True # 设为 True 更符合直觉
            ) for _ in range(num_layers)
        ])
        
        self.final_layer_norm = nn.LayerNorm(hidden_dim)

        # --- 输出头 ---
        self.rates_out = nn.Sequential(nn.Linear(hidden_dim, 3))
        self.ins_logits_out = nn.Linear(hidden_dim, vocab_size)
        self.sub_logits_out = nn.Linear(hidden_dim, vocab_size)

        # 初始化 time projection 的权重和偏置为0
        nn.init.zeros_(self.time_proj[-1].weight)
        nn.init.zeros_(self.time_proj[-1].bias)

    def forward(
        self, 
        tokens: torch.Tensor,         # x_t (正在生成的CDR3) [B, L]
        VDJ_embeddings: torch.Tensor, # [B, 3, VDJ_embedding_dim]
        time_step: torch.Tensor,      # t [B, 1]
        padding_mask: torch.Tensor,   # x_t 的 mask [B, L]，True 表示 padding
    ):
        batch_size, x_len = tokens.shape
        device = tokens.device  # 获取模型所在的设备

        # 处理主序列 x_t
        token_emb = self.token_embedding(tokens)  # [B, L, H]
        dtype = token_emb.dtype  # 获取模型的数据类型（可能是 float32 或 bfloat16）
        
        # VDJ embeddings 维度转换 [B, 3, VDJ_embedding_dim] -> [B, 3, hidden_dim]
        VDJ_embeddings = VDJ_embeddings.to(dtype=dtype)
        vdj_embs = self.VDJ_layers(VDJ_embeddings) # [B, 3, hidden_dim]
        vdj_embs = F.layer_norm(vdj_embs, (self.hidden_dim,))
        
        # 将 VDJ embeddings 作为 prefix 拼接到 token_emb 前面
        # vdj_embs: [B, 3, H], token_emb: [B, L, H] -> concat: [B, 3+L, H]
        x_concat = torch.cat([vdj_embs, token_emb], dim=1)  # [B, 3+L, H]
        total_len = 3 + x_len  # 总长度：3 (VDJ) + L (tokens)
        
        # Time embedding: [B, H] -> [B, 1, H] -> [B, 3+L, H]
        time_emb = self.time_embedding(time_step)  # [B, H]
        
        # Position embedding: 直接从 0 到 total_len-1
        pos_indices = torch.arange(total_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)  # [B, 3+L]
        pos_emb = self.pos_embedding(pos_indices)  # [B, 3+L, H]
        
        # 核心输入融合：VDJ+Token + Position
        x = x_concat + pos_emb  # [B, 3+L, H]
        
        # 扩展 padding_mask: 前3位（VDJ embeddings）设为False（不被mask）
        vdj_padding_mask = torch.zeros((batch_size, 3), device=device, dtype=torch.bool)  # [B, 3]
        full_padding_mask = torch.cat([vdj_padding_mask, padding_mask], dim=1)  # [B, 3+L]
        
        # Transformer Encoder 层 (Self-Attention)
        t_params = self.time_proj(time_emb).view(batch_size, self.num_layers, 2, self.hidden_dim)
        for i, layer in enumerate(self.layers):
            s = t_params[:, i, 0, :].unsqueeze(1)
            b = t_params[:, i, 1, :].unsqueeze(1)
            x = x * (1 + s) + b
            x = layer(x, src_key_padding_mask=full_padding_mask)

        x = self.final_layer_norm(x)
        
        # 只取 tokens 对应的部分（排除前3个VDJ embeddings位置）
        x_tokens = x[:, 3:, :]  # [B, L, H]
        
        # 计算输出（只对 tokens 部分）
        rates = F.softplus(self.rates_out(x_tokens))
        ins_probs = F.softmax(self.ins_logits_out(x_tokens), dim=-1)
        sub_probs = F.softmax(self.sub_logits_out(x_tokens), dim=-1)

        # Padding 处理
        mask_expanded = (~padding_mask).unsqueeze(-1).float()
        return rates * mask_expanded, ins_probs * mask_expanded, sub_probs * mask_expanded