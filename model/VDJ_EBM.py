import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

class VDJEnergyModel(nn.Module):
    def __init__(self, evo2_dim=1280, hidden_dim=256):
        super().__init__()
        # 降维层：将 EVO2 的高维向量映射到紧凑特征空间
        self.v_proj = nn.Linear(evo2_dim, hidden_dim)
        self.d_proj = nn.Linear(evo2_dim, hidden_dim)
        self.j_proj = nn.Linear(evo2_dim, hidden_dim)
        
        # 能量评分网络：学习 V-D-J 的相容性
        self.energy_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # 输出标量能量值
        )

    def forward(self, vdj_embeddings: torch.Tensor):
        """
        Args:
            vdj_embeddings: Tensor [batch, 3, evo2_dim] - V, D, J embeddings
        Returns:
            energy: Tensor [batch, 1] - 能量值
        """
        # 分离 V, D, J embeddings
        v_emb = vdj_embeddings[:, 0, :]  # [batch, evo2_dim]
        d_emb = vdj_embeddings[:, 1, :]  # [batch, evo2_dim]
        j_emb = vdj_embeddings[:, 2, :]  # [batch, evo2_dim]
        
        # 1. 投影特征
        v_feat = self.v_proj(v_emb)
        d_feat = self.d_proj(d_emb)
        j_feat = self.j_proj(j_emb)
        
        # 2. 拼接特征
        combined = torch.cat([v_feat, d_feat, j_feat], dim=-1)
        
        # 3. 计算能量值
        energy = self.energy_net(combined)
        return energy

# 训练与采样管理器
class VDJManager:
    def __init__(self, model, vdj_embeddings_dict: dict, device: Optional[torch.device] = None):
        """
        Args:
            model: VDJEnergyModel 实例
            vdj_embeddings_dict: 字典，键为 "V_{v_call}", "D_{d_call}", "J_{j_call}"，值为 embedding tensor
            device: 计算设备
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 从字典中提取所有 V, D, J 基因名称和对应的 embeddings
        v_keys = sorted([k for k in vdj_embeddings_dict.keys() if k.startswith('V_')])
        d_keys = sorted([k for k in vdj_embeddings_dict.keys() if k.startswith('D_')])
        j_keys = sorted([k for k in vdj_embeddings_dict.keys() if k.startswith('J_')])
        
        self.v_keys = v_keys
        self.d_keys = d_keys
        self.j_keys = j_keys
        
        # 构建所有 V, D, J embeddings (取平均，与 dataloader 保持一致)
        self.v_embs = torch.stack([
            vdj_embeddings_dict[k].mean(dim=0) if vdj_embeddings_dict[k].dim() > 1 
            else vdj_embeddings_dict[k] 
            for k in v_keys
        ]).to(self.device)  # [num_v, evo2_dim]
        
        self.d_embs = torch.stack([
            vdj_embeddings_dict[k].mean(dim=0) if vdj_embeddings_dict[k].dim() > 1 
            else vdj_embeddings_dict[k] 
            for k in d_keys
        ]).to(self.device)  # [num_d, evo2_dim]
        
        self.j_embs = torch.stack([
            vdj_embeddings_dict[k].mean(dim=0) if vdj_embeddings_dict[k].dim() > 1 
            else vdj_embeddings_dict[k] 
            for k in j_keys
        ]).to(self.device)  # [num_j, evo2_dim]
        
        # 事先计算所有可能的组合索引 (Cartesian product)
        self.all_combinations = torch.cartesian_prod(
            torch.arange(len(v_keys)),
            torch.arange(len(d_keys)),
            torch.arange(len(j_keys))
        ).to(self.device)
        
        # 预计算所有组合的 embeddings (用于快速采样)
        # 如果组合太多，可以延迟计算或使用缓存
        self._all_comb_embs = None
        self._all_energies_cache = None

    def train_step(self, vdj_embeddings: torch.Tensor, optimizer):
        """
        使用负对数似然 (NLL) 训练。
        
        Args:
            vdj_embeddings: Tensor [batch, 3, evo2_dim] - 真实样本的 VDJ embeddings
            optimizer: 优化器
        Returns:
            loss: 损失值
        """
        optimizer.zero_grad()
        
        # 获取模型参数的 dtype，确保输入数据与模型参数 dtype 一致
        model_dtype = next(self.model.parameters()).dtype
        vdj_embeddings = vdj_embeddings.to(dtype=model_dtype)
        
        # 1. 计算真实样本的能量
        real_energy = self.model(vdj_embeddings)  # [batch, 1]
        
        # 2. 计算所有组合的能量（全空间计算）
        if self._all_comb_embs is None:
            # 构建所有组合的 embeddings
            all_v = self.v_embs[self.all_combinations[:, 0]].to(dtype=model_dtype)
            all_d = self.d_embs[self.all_combinations[:, 1]].to(dtype=model_dtype)
            all_j = self.j_embs[self.all_combinations[:, 2]].to(dtype=model_dtype)
            self._all_comb_embs = torch.stack([all_v, all_d, all_j], dim=1)  # [num_all_comb, 3, evo2_dim]
        
        all_energies = self.model(self._all_comb_embs).squeeze()  # [num_all_comb]
        
        # 损失函数：真实样本的分数要在全空间中最大化 (即能量最小化)
        # P(x) = exp(-E_real) / sum(exp(-E_all))
        # LogP(x) = -E_real - log(sum(exp(-E_all)))
        log_z = torch.logsumexp(-all_energies, dim=0)
        loss = torch.mean(real_energy) + log_z
        
        loss.backward()
        optimizer.step()
        return loss.item()

    @torch.no_grad()
    def sample(self, num_samples: int = 1, temperature: float = 1.0) -> List[Tuple[str, str, str]]:
        """
        从学习到的分布中采样
        
        Args:
            num_samples: 采样数量
            temperature: 温度参数，控制采样分布的尖锐程度
        Returns:
            List of (v_call, d_call, j_call) tuples
        """
        # 获取模型参数的 dtype
        model_dtype = next(self.model.parameters()).dtype
        
        # 如果组合太多，使用缓存或分批计算
        if self._all_energies_cache is None or not self.model.training:
            if self._all_comb_embs is None:
                all_v = self.v_embs[self.all_combinations[:, 0]].to(dtype=model_dtype)
                all_d = self.d_embs[self.all_combinations[:, 1]].to(dtype=model_dtype)
                all_j = self.j_embs[self.all_combinations[:, 2]].to(dtype=model_dtype)
                self._all_comb_embs = torch.stack([all_v, all_d, all_j], dim=1)
            
            # 计算所有组合的能量
            energies = self.model(self._all_comb_embs).squeeze()  # [num_all_comb]
            self._all_energies_cache = energies
        
        # 计算概率分布
        logits = -self._all_energies_cache / temperature
        probs = F.softmax(logits, dim=0)
        
        # 采样
        indices = torch.multinomial(probs, num_samples, replacement=True)
        sampled_comb = self.all_combinations[indices]
        
        # 转换为基因名称
        results = []
        for idx in sampled_comb:
            v_idx, d_idx, j_idx = idx[0].item(), idx[1].item(), idx[2].item()
            v_call = self.v_keys[v_idx].replace('V_', '')
            d_call = self.d_keys[d_idx].replace('D_', '')
            j_call = self.j_keys[j_idx].replace('J_', '')
            results.append((v_call, d_call, j_call))
        
        return results