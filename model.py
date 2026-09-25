"""
从0到1实现Transformer架构

Assembled from your step-by-step solutions.
"""

# Step 1 - bpe_tokenize
from typing import List, Tuple

def bpe_tokenize(text: str, merges: List[Tuple[str, str]]) -> List[str]:
    # 建立合并规则 -> 优先级的映射，越靠前优先级越小
    merge_rank = {}
    for idx, (a, b) in enumerate(merges):
        merge_rank[(a, b)] = idx
    
    # 按空格切分出单词
    words = text.split()
    result = []
    
    for word in words:
        # 单词拆成单个字符，末尾加上 </w>
        tokens = list(word) + ['</w>']
        
        while True:
            # 找出当前所有相邻token对，筛选存在于merge_rank中的
            pairs = []
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                if pair in merge_rank:
                    pairs.append((merge_rank[pair], i, pair))
            if not pairs:
                break
            
            # 选优先级最高（rank最小）的一对
            pairs.sort()
            best_rank, best_i, best_pair = pairs[0]
            
            # 执行合并，从左到右合并所有匹配best_pair的位置
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens)-1 and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(tokens[i] + tokens[i+1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        result.extend(tokens)
    return result

# Step 2 - build_token_id_matrix
from typing import List, Dict

def build_token_id_matrix(
    batch_tokens: List[List[str]],
    vocab: Dict[str, int],
    max_len: int
) -> List[List[int]]:
    matrix = []
    pad_id = vocab['<pad>']
    unk_id = vocab['<unk>']
    bos_id = vocab['<bos>']
    eos_id = vocab['<eos>']
    
    for tokens in batch_tokens:
        # 1. 前后加上 <bos> 和 <eos>
        seq = ['<bos>'] + tokens + ['<eos>']
        # 2. token转id，未知token替换为<unk>
        ids = [vocab.get(tok, unk_id) for tok in seq]
        # 3. 截断到max_len
        ids = ids[:max_len]
        # 4. 末尾补<pad>直到长度等于max_len
        pad_needed = max_len - len(ids)
        ids += [pad_id] * pad_needed
        matrix.append(ids)
    return matrix

# Step 3 - __init__
import math
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.lut = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 1. self.lut(token_ids): 查表取出词向量，形状由 (B, L) 变为 (B, L, d_model)
        # 2. 乘以 math.sqrt(self.d_model) 缩放尺度
        return self.lut(token_ids) * math.sqrt(self.d_model)

# Step 4 - sinusoidal_encoding
import math
import torch

def sinusoidal_encoding(max_len: int, d_model: int, device=None) -> torch.Tensor:
    # pos: [max_len, 1]
    pos = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
    # 取 i 为 0,1,...,d_model//2 -1
    i = torch.arange(0, d_model // 2, dtype=torch.float32, device=device)
    omega = torch.pow(10000.0, (-2 * i) / d_model)

    pe = torch.zeros((max_len, d_model), dtype=torch.float32, device=device)
    pe[:, 0::2] = torch.sin(pos * omega)
    pe[:, 1::2] = torch.cos(pos * omega)
    return pe

# Step 5 - __init__
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 1. 初始化 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        
        # 2. 生成位置索引 pos: [0, 1, 2, ..., max_len - 1]，形状 (max_len, 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 3. 计算衰减项 div_term = 10000^(-2i / d_model)
        # 用 exp(log(...)) 提高数值计算稳定性
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 4. 偶数维度填 sin，奇数维度填 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 5. 扩充 batch 维度为 (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        
        # 6. 注册为 buffer：随模型自动切换 CPU/GPU，但不会被当作待学习参数更新梯度
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x 形状: (B, L, d_model)
        # 截取前 L 个位置编码，利用广播机制自动与 B 个样本逐元素相加
        x = x + self.pe[:, :x.size(1)]
        # 应用 Dropout 并返回
        return self.dropout(x)

# Step 6 - make_src_mask
import math
import torch
import torch.nn as nn
from typing import Optional, Any
def make_src_mask(src_ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    # 1. (src_ids != pad_id): 非 pad 设为 True，pad 设为 False，形状为 (B, L)
    # 2. .unsqueeze(1).unsqueeze(2): 依次在第 1 维和第 2 维插入单维度，形状变为 (B, 1, 1, L)
    return (src_ids != pad_id).unsqueeze(1).unsqueeze(2)

# Step 7 - subsequent_mask
import torch

def subsequent_mask(
    size: int,
    device = None,
) -> torch.Tensor:
    # 1. torch.ones((size, size), dtype=torch.bool, device=device): 创建布尔矩阵
    # 2. torch.tril(...): 取下三角矩阵（主对角线及下方为 True，上方为 False）
    # 3. .unsqueeze(0).unsqueeze(0): 扩充 batch 维度和 head 维度，变为 (1, 1, size, size)
    mask = torch.tril(torch.ones((size, size), dtype=torch.bool, device=device))
    return mask.unsqueeze(0).unsqueeze(0)

# Step 8 - make_tgt_mask (not yet solved)
# TODO: implement

# Step 9 - shift_targets_right
import torch

def shift_targets_right(
    target_ids: torch.Tensor,
    bos_id: int,
) -> torch.Tensor:
    # 1. 创建同形状、同设备、同 dtype 的新张量，确保独立于原输入
    shifted = torch.empty_like(target_ids)
    
    # 2. 第 0 列全部填充起始标记 bos_id
    shifted[:, 0] = bos_id
    
    # 3. 当序列长度 L > 1 时，将原序列除最后一列外的切片 [:, :-1] 复制到右侧 [:, 1:]
    if target_ids.size(1) > 1:
        shifted[:, 1:] = target_ids[:, :-1]
        
    return shifted

# Step 10 - __init__
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # q/k/v 形状: [B, H, S, D]
        d_k = q.size(-1)

        # 1. 计算点积注意力得分并缩放: (Q @ K^T) / sqrt(d_k)
        # 结果形状: [B, H, S_q, S_k]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # 2. 掩码操作 (如果有 mask，将 mask 为 0 或 False 的无效位置填充为 -inf 或 -1e9)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 3. Softmax 归一化为概率分布
        attn_weights = F.softmax(scores, dim=-1)

        # 4. 对注意力权重应用 Dropout 并与 V 相乘加权求和
        # output 形状: [B, H, S, D]
        output = torch.matmul(self.dropout(attn_weights), v)

        return output, attn_weights

# Step 11 - __init__
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# 基础零件: 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout_p: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attn_weights), v)
        return output, attn_weights


# 完整模块: 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim 必须能被 num_heads 整除"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 1. 按照注释要求定义 4 个线性投影矩阵 W_q, W_k, W_v, W_o
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # 兼容小写命名的测试用例
        self.w_q = self.W_q
        self.w_k = self.W_k
        self.w_v = self.W_v
        self.w_o = self.W_o

        # 2. 引入上方的缩放点积注意力积木
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x 形状: [B, S, embed_dim]
        B, S, _ = x.shape

        # 1. 线性投影并切分成多头: [B, S, embed_dim] -> [B, S, H, D] -> [B, H, S, D]
        q = self.W_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # 2. 调用缩放点积注意力计算
        # out 形状: [B, H, S, head_dim]
        out, _ = self.attention(q, k, v, mask=mask)

        # 3. 拼合多头: [B, H, S, head_dim] -> [B, S, H, head_dim] -> [B, S, embed_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)

        # 4. 经过输出层 W_o 融合多头特征
        return self.W_o(out)

# Step 12 - __init__
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, model_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # gamma初始化为1，beta初始化为0，shape [model_dim]
        self.gamma = nn.Parameter(torch.ones(model_dim))
        self.beta = nn.Parameter(torch.zeros(model_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, model_dim]，在最后一维model_dim做归一化
        mean = x.mean(dim=-1, keepdim=True)
        # 总体方差 unbiased=False，除以N
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / torch.sqrt(var + self.eps)
        out = normalized * self.gamma + self.beta
        return out

# Step 13 - __init__
import torch
import torch.nn as nn
import torch.nn.functional as F

class FFN(nn.Module):
    def __init__(self, model_dim: int, intermediate_dim: int):
        super().__init__()
        # 1. 升维线性层: model_dim -> intermediate_dim
        self.w_up = nn.Linear(model_dim, intermediate_dim)
        # 2. 降维线性层: intermediate_dim -> model_dim
        self.w_down = nn.Linear(intermediate_dim, model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先升维映射 -> ReLU 激活截断负数 -> 降维映射回原维度
        return self.w_down(F.relu(self.w_up(x)))

# Step 14 - encoder_layer_forward (not yet solved)
# TODO: implement

# Step 15 - __init__ (not yet solved)
# TODO: implement

# Step 16 - decoder_layer_forward (not yet solved)
# TODO: implement

# Step 17 - __init__ (not yet solved)
# TODO: implement

# Step 18 - __init__ (not yet solved)
# TODO: implement

# Step 19 - __init__ (not yet solved)
# TODO: implement

# Step 20 - tie_target_embedding (not yet solved)
# TODO: implement

# Step 21 - __init__ (not yet solved)
# TODO: implement

# Step 22 - label_smoothing_distribution (not yet solved)
# TODO: implement

# Step 23 - loss_ignoring_pad (not yet solved)
# TODO: implement

# Step 24 - __init__ (not yet solved)
# TODO: implement

# Step 25 - noam_learning_rate (not yet solved)
# TODO: implement

# Step 26 - make_optimizer (not yet solved)
# TODO: implement

# Step 27 - optimizer_hyperparameters (not yet solved)
# TODO: implement

# Step 28 - transformer_training_loss (not yet solved)
# TODO: implement

# Step 29 - backward_step (not yet solved)
# TODO: implement

# Step 30 - train_batch (not yet solved)
# TODO: implement

# Step 31 - evaluate_batch (not yet solved)
# TODO: implement

# Step 32 - checkpoint_roundtrip (not yet solved)
# TODO: implement

# Step 33 - greedy_next_token (not yet solved)
# TODO: implement

# Step 34 - greedy_decode (not yet solved)
# TODO: implement

# Step 35 - greedy_decode_eos (not yet solved)
# TODO: implement

# Step 36 - beam_expand_scores (not yet solved)
# TODO: implement

# Step 37 - beam_topk (not yet solved)
# TODO: implement

# Step 38 - update_finished_beams (not yet solved)
# TODO: implement

# Step 39 - length_penalty (not yet solved)
# TODO: implement

# Step 40 - beam_decode_step (not yet solved)
# TODO: implement

# Step 41 - beam_decode (not yet solved)
# TODO: implement

# Step 42 - tiny_model_inference (not yet solved)
# TODO: implement

# Step 43 - end_to_end_decode (not yet solved)
# TODO: implement
