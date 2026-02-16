#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import pyarrow.dataset as ds
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Tuple, Optional
from utils.dataloader_utils import opt_align_xs_to_zs

# --- SimpleTokenizer 保持不变 ---
class SimpleTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.token_to_id = vocab
        self.id_to_token = {idx: token for token, idx in vocab.items()}
        self.actual_vocab_size = len(vocab)
        self.pad_token_id, self.bos_token_id, self.gap_token_id = self._get_special_tokens()
        self.vocab_size = self.actual_vocab_size + 3
    
    def _get_special_tokens(self) -> Tuple[int, int, int]:
        return self.actual_vocab_size, self.actual_vocab_size + 1, self.actual_vocab_size + 2
    
    def encode(self, text: str, convert_to_tensor: bool = False) -> List[int]:
        tokens = [self.token_to_id[char] for char in text if char in self.token_to_id]
        if convert_to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long)
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        return ''.join(self.id_to_token.get(tid, '') for tid in token_ids)

# --- ParquetStreamingDataset ---
class ParquetStreamingDataset(IterableDataset):
    def __init__(self, 
        base_dir: str, 
        VDJ_cols: list, 
        VDJ_germline_ids_col: list,
        seq_col: str, 
        shuffle_files: bool = True,
        buffer_size: int = 10000  # 随机缓冲区大小
    ):
        self.base_dir = base_dir
        self.VDJ_cols = VDJ_cols
        self.VDJ_germline_ids_col = VDJ_germline_ids_col
        self.seq_col = seq_col
        self.shuffle_files = shuffle_files
        self.buffer_size = buffer_size
        # 加载数据集索引，这不会读取实际数据内容
        self.dataset = ds.dataset(self.base_dir, format="parquet", exclude_invalid_files=True)
        self.fragments = list(self.dataset.get_fragments())
    
    def _get_worker_fragments(self):
        """处理 DDP 和 Multi-worker 的分片逻辑"""
        if dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0
        
        # GPU 级别分片
        rank_fragments = self.fragments[rank::world_size]
        
        # Worker 级别分片
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return rank_fragments, rank, 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            return rank_fragments[worker_id::num_workers], rank, worker_id

    def __iter__(self):
        worker_fragments, rank, worker_id = self._get_worker_fragments()
        
        # 初始 Buffer
        buffer = []
        
        # --- 无限循环：实现多 Epoch ---
        while True:
            if self.shuffle_files:
                random.shuffle(worker_fragments)
            
            for fragment in worker_fragments:
                # 使用 scanner 进行流式读取，避免使用 fragment.to_table()
                # 这样即使 Parquet 文件很大，内存占用也很稳定
                scanner = fragment.scanner(columns=self.VDJ_cols + self.VDJ_germline_ids_col + [self.seq_col])
                
                for batch in scanner.to_batches():
                    # 将 pyarrow.RecordBatch 转换为 Python 列表以供处理
                    # process_batch 逻辑直接内联以提高效率
                    col_data = [batch.column(c).to_pylist() for c in self.VDJ_cols + self.VDJ_germline_ids_col + [self.seq_col]]
                    samples = list(zip(*col_data))
                    
                    for sample in samples:
                        if len(buffer) < self.buffer_size:
                            buffer.append(sample)
                        else:
                            # 缓冲区满了，随机替换一个并 yield
                            idx = random.randint(0, self.buffer_size - 1)
                            yield_sample = buffer[idx]
                            buffer[idx] = sample
                            yield yield_sample
            
            # 如果跑完了一轮还没填满 buffer（数据量极小的情况）
            if not worker_fragments:
                break # 防止死循环

# --- OAS_dataloader 保持不变，但增加 max_seq_len 处理逻辑 ---
class OAS_dataloader:
    def __init__(self, 
        base_dir, 
        VDJ_cols, 
        VDJ_germline_ids_col, 
        seq_col, 
        vdj_embeddings_dict,
        tokenizer, 
        max_seq_len, 
        shuffle_files=True, 
        buffer_size=10000,
        random_amino_acid_num=10):
        self.vdj_embeddings_dict = vdj_embeddings_dict
        self.train_dataset = ParquetStreamingDataset(
            base_dir=base_dir, 
            VDJ_cols=VDJ_cols, 
            VDJ_germline_ids_col=VDJ_germline_ids_col,
            seq_col=seq_col,
            shuffle_files=shuffle_files,
            buffer_size=buffer_size
        )
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.random_amino_acid_num = random_amino_acid_num

    def batch_collate(self, batch) -> Tuple:
        pad_id = self.tokenizer.pad_token_id
        bos_id = self.tokenizer.bos_token_id
        gap_id = self.tokenizer.gap_token_id

        # 解包 batch (V, D, J, Seq)
        v_seq, d_seq, j_seq, v_call, d_call, j_call, anti_seq = zip(*batch)

        # v_end_j_start: 在 v_end 和 j_start 之间插入random_amino_acid_num个随机氨基酸token
        amino_acid_token_ids = list(range(self.tokenizer.actual_vocab_size))  # 0到19，共20个氨基酸
        x_0_list = []
        for v, j in zip(v_seq, j_seq):
            v_end = v[-6:]  # v序列的最后6个字符
            j_start = j[:6]  # j序列的前6个字符
            # 分别tokenize v_end和j_start
            v_end_tokens = self.tokenizer.encode(v_end, convert_to_tensor=True)
            j_start_tokens = self.tokenizer.encode(j_start, convert_to_tensor=True)
            # 生成random_amino_acid_num个随机氨基酸token IDs
            random_token_ids = torch.tensor(
                random.choices(amino_acid_token_ids, k=self.random_amino_acid_num), 
                dtype=torch.long
            )
            # 连接: v_end + random_amino_acid_num个随机token + j_start
            x_0_tokens = torch.cat([v_end_tokens, random_token_ids, j_start_tokens], dim=0)
            x_0_list.append(x_0_tokens)
            
        # vdj_embeddings
        vdj_embeddings = []
        for v_call, d_call, j_call in zip(v_call, d_call, j_call):
            v_emb = self.vdj_embeddings_dict[f"V_{v_call}"].mean(dim=0)
            d_emb = self.vdj_embeddings_dict[f"D_{d_call}"].mean(dim=0)
            j_emb = self.vdj_embeddings_dict[f"J_{j_call}"].mean(dim=0)
            vdj_embeddings.append(torch.stack([v_emb, d_emb, j_emb], dim=0))
        vdj_embeddings = torch.stack(vdj_embeddings, dim=0)
        
        # CDR3
        x_1_list = []
        for v_seq, d_seq, j_seq, anti_seq in zip(v_seq, d_seq, j_seq, anti_seq):
            CDR3_start = len(v_seq) - 6
            CDR3_end = len(anti_seq) - len(j_seq) + 6
            CDR3 = anti_seq[CDR3_start:CDR3_end]
            x_1_list.append(self.tokenizer.encode(CDR3, convert_to_tensor=True))
        
        # sequence alignment
        z_1_list, z_0_list = [], []
        for i in range(len(x_1_list)):
            _z0, _z1 = opt_align_xs_to_zs(x_0_list[i].unsqueeze(0), x_1_list[i].unsqueeze(0), gap_token=gap_id)
            z_0_list.append(_z0.squeeze(0))
            z_1_list.append(_z1.squeeze(0))

        # 动态截断与 Padding
        def pad_and_stack(tensors, max_len):
            # 考虑 BOS 占位，预留长度
            actual_max = min(max_len, self.max_seq_len - 1) if self.max_seq_len else max_len
            processed = []
            for t in tensors:
                t = t[:actual_max] # 截断
                p = F.pad(t, (0, actual_max - t.shape[0]), value=pad_id) # 填充
                p = F.pad(p, (1, 0), value=bos_id) # 加 BOS
                processed.append(p)
            return torch.stack(processed, dim=0).long()

        z_max = max(len(z) for z in z_0_list)
        x0_max = max(len(x) for x in x_0_list)
        x1_max = max(len(x) for x in x_1_list)

        x_0 = pad_and_stack(x_0_list, x0_max)
        x_1 = pad_and_stack(x_1_list, x1_max)
        z_0 = pad_and_stack(z_0_list, z_max)
        z_1 = pad_and_stack(z_1_list, z_max)
        
        t = torch.rand(len(batch), 1)
        padding_mask = (x_1 == pad_id)
        
        return x_0, x_1, z_0, z_1, t, padding_mask, vdj_embeddings

    def get_loader(self, batch_size: int, num_workers: int, prefetch_factor: int) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            collate_fn=self.batch_collate,
            pin_memory=True # 推荐开启以加速 Tensor 传输到 GPU
        )

# --- 使用示例 ---
if __name__ == "__main__":
    import time
    import json

    data_path = "/inspire/ocean/project/ai3-lab/public/lbh/codebase/dfm/VDJ_flows_v2/data/" 
    batch_size = 64
    prefetch_factor = 2
    num_workers = 4
    VDJ_cols = ["v_germline", "d_germline", "j_germline"]
    VDJ_germline_ids_col = ["v_call", "d_call", "j_call"]
    vdj_embeddings_dict_path = os.path.join(data_path, "germline_embeddings.pt")
    vdj_embeddings_dict = torch.load(vdj_embeddings_dict_path)
    seq_col = "sequence_alignment_aa_filled"
    max_seq_len = 192

    # 1. 初始化 Tokenizer
    vocab_path = "/inspire/ocean/project/ai3-lab/public/lbh/codebase/dfm/VDJ-flows-HF/data/protein_one_letter_vocab.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    tokenizer = SimpleTokenizer(vocab)
    
    # 2. 实例化
    dataloader = OAS_dataloader(
        base_dir=data_path,
        VDJ_cols=VDJ_cols,
        seq_col=seq_col,
        VDJ_germline_ids_col=VDJ_germline_ids_col,
        vdj_embeddings_dict=vdj_embeddings_dict,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        shuffle_files=True
    )
    train_loader = dataloader.get_loader(batch_size, num_workers, prefetch_factor)
    # 3. 运行示例循环
    print("\n--- 开始模拟 DataLoader 迭代 (训练循环) ---")
    for step, (x_0_tokens, x_1_tokens, z_0_tokens, z_1_tokens, t, padding_mask, vdj_germline_ids) in enumerate(train_loader):
        if step > 5:
            break