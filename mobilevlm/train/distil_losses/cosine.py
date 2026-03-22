"""
Cosine distance-based distillation loss for knowledge distillation.

提供两种Cosine实现：
1. Cosine: 在logits空间计算，先标准化（零均值、单位方差）再计算cosine distance
2. CosineProbs: 在概率空间计算，先Softmax再计算cosine distance（推荐）

相比MSE，cosine distance只关注分布的形状和方向，而忽略绝对值的大小。

优势：
1. Scale-free: 不受teacher logits幅值变化的影响
2. 关注分布形状: 只学习各token之间的相对关系
3. 训练更稳定: 避免追逐绝对数值导致的不稳定
4. Standardization: Cosine版本通过标准化进一步消除均值和方差的影响
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def cosine_distance(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    在logits空间计算cosine distance（scale-free + standardization）。
    
    先对logits进行标准化（零均值、单位方差），再计算cosine distance。
    Cosine distance = 1 - cosine_similarity
    只关注logits向量的分布形状，忽略均值、方差和幅值的影响。
    
    Args:
        logits: 学生模型的logits [batch, seq_len, vocab_size]
        teacher_logits: 教师模型的logits [batch, seq_len, vocab_size]
        mask: 有效位置的mask [batch, seq_len]
        temperature: 温度参数，用于缩放logits（默认1.0）
    
    Returns:
        Cosine distance损失值（标量）
    """
    # 确保所有张量的序列长度一致
    min_seq_len = min(logits.shape[1], teacher_logits.shape[1], mask.shape[1])
    logits = logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    mask = mask[:, :min_seq_len]
    
    # 检查nan和inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("[Cosine] Warning: Student logits contain NaN or Inf values")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        print("[Cosine] Warning: Teacher logits contain NaN or Inf values")
        return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    
    # 应用温度缩放（可选，对cosine distance影响较小，因为scale-free）
    if temperature != 1.0:
        logits = logits / temperature
        teacher_logits = teacher_logits / temperature
    
    # 标准化处理：(x - mean) / std，使logits具有零均值和单位方差
    # 这样可以进一步消除均值和方差的影响，只关注分布形状
    logits_mean = logits.mean(dim=-1, keepdim=True)
    logits_std = logits.std(dim=-1, keepdim=True) + 1e-8  # 添加小常数避免除零
    logits_normalized = (logits - logits_mean) / logits_std
    
    teacher_logits_mean = teacher_logits.mean(dim=-1, keepdim=True)
    teacher_logits_std = teacher_logits.std(dim=-1, keepdim=True) + 1e-8
    teacher_logits_normalized = (teacher_logits - teacher_logits_mean) / teacher_logits_std
    
    # 检查标准化后的结果
    if torch.isnan(logits_normalized).any() or torch.isinf(logits_normalized).any():
        print("[Cosine] Warning: Normalized student logits contain NaN or Inf values")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if torch.isnan(teacher_logits_normalized).any() or torch.isinf(teacher_logits_normalized).any():
        print("[Cosine] Warning: Normalized teacher logits contain NaN or Inf values")
        return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    
    # 计算cosine similarity在vocab维度 [batch, seq_len, vocab_size] -> [batch, seq_len]
    # F.cosine_similarity默认在dim=-1计算
    # cosine_sim = (A·B) / (||A|| * ||B||)
    cosine_sim = F.cosine_similarity(logits_normalized, teacher_logits_normalized, dim=-1)
    
    # Cosine distance = 1 - cosine_similarity
    # 范围: [0, 2]，其中0表示完全相同，2表示完全相反
    cosine_dist = 1.0 - cosine_sim  # [batch, seq_len]
    
    # 使用mask加权平均
    mask_sum = mask.sum()
    if mask_sum == 0:
        print("[Cosine] Warning: mask_sum is 0, no valid positions for distillation")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    loss = (cosine_dist * mask).sum() / mask_sum
    
    # 调试信息：打印损失值和相关统计
    print(f"[Cosine DEBUG] loss: {loss.item():.6f}, mask_sum: {mask_sum.item():.1f}, "
          f"cosine_sim range: [{cosine_sim.min().item():.6f}, {cosine_sim.max().item():.6f}], "
          f"cosine_dist range: [{cosine_dist.min().item():.6f}, {cosine_dist.max().item():.6f}], "
          f"normalized logits std: [{logits_std.mean().item():.6f}]")
    
    # 检查最终结果是否为nan/inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[Cosine] Warning: Final loss is NaN or Inf")
        print(f"  - logits range: [{logits.min():.4f}, {logits.max():.4f}]")
        print(f"  - teacher_logits range: [{teacher_logits.min():.4f}, {teacher_logits.max():.4f}]")
        print(f"  - cosine_sim range: [{cosine_sim.min():.4f}, {cosine_sim.max():.4f}]")
        print(f"  - mask_sum: {mask_sum}")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    return loss


def cosine_probs(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    在概率空间计算cosine distance（推荐方法）。
    
    将logits转换为概率分布后计算cosine distance，更稳定。
    最小化cosine distance = 最大化cosine similarity
    
    Args:
        logits: 学生模型的logits [batch, seq_len, vocab_size]
        teacher_logits: 教师模型的logits [batch, seq_len, vocab_size]
        mask: 有效位置的mask [batch, seq_len]
        temperature: 温度参数，用于软化概率分布（默认1.0）
    
    Returns:
        Cosine distance损失值（标量）
    """
    # 确保所有张量的序列长度一致
    min_seq_len = min(logits.shape[1], teacher_logits.shape[1], mask.shape[1])
    logits = logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    mask = mask[:, :min_seq_len]
    
    # 检查nan和inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("[CosineProbs] Warning: Student logits contain NaN or Inf values")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        print("[CosineProbs] Warning: Teacher logits contain NaN or Inf values")
        return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    
    # 转换为概率分布（使用温度缩放）
    student_probs = F.softmax(logits / temperature, dim=-1, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    
    # 处理inf值（将其概率设为0）
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    student_probs = torch.masked_fill(student_probs, inf_mask, 0.0)
    teacher_probs = torch.masked_fill(teacher_probs, inf_mask, 0.0)
    
    # 计算cosine similarity在vocab维度 [batch, seq_len, vocab_size] -> [batch, seq_len]
    # F.cosine_similarity默认在dim=-1计算
    # cosine_sim = (A·B) / (||A|| * ||B||)
    cosine_sim = F.cosine_similarity(student_probs, teacher_probs, dim=-1)
    
    # Cosine distance = 1 - cosine_similarity
    # 范围: [0, 2]，其中0表示完全相同，2表示完全相反
    cosine_dist = 1.0 - cosine_sim  # [batch, seq_len]
    
    # 使用mask加权平均
    mask_sum = mask.sum()
    if mask_sum == 0:
        print("[CosineProbs] Warning: mask_sum is 0, no valid positions for distillation")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    loss = (cosine_dist * mask).sum() / mask_sum
    
    # 调试信息：打印损失值和相关统计
    print(f"[CosineProbs DEBUG] loss: {loss.item():.6f}, mask_sum: {mask_sum.item():.1f}, "
          f"cosine_sim range: [{cosine_sim.min().item():.6f}, {cosine_sim.max().item():.6f}], "
          f"cosine_dist range: [{cosine_dist.min().item():.6f}, {cosine_dist.max().item():.6f}]")
    
    # 如果使用了温度缩放，可以选择调整损失尺度
    # 注意：在概率空间，温度缩放的影响已经体现在概率分布中
    # 通常不需要像KD那样乘以temperature^2，但可以根据实验调整
    if temperature != 1.0:
        loss = loss * (temperature ** 2)
    
    # 检查最终结果是否为nan/inf
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"[CosineProbs] Warning: Final loss is NaN or Inf")
        print(f"  - student_probs range: [{student_probs.min():.4f}, {student_probs.max():.4f}]")
        print(f"  - teacher_probs range: [{teacher_probs.min():.4f}, {teacher_probs.max():.4f}]")
        print(f"  - cosine_sim range: [{cosine_sim.min():.4f}, {cosine_sim.max():.4f}]")
        print(f"  - mask_sum: {mask_sum}")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    return loss


class Cosine(DistilLoss):
    """Cosine distance蒸馏损失（scale-free + standardization，只关注分布形状）"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return cosine_distance(logits, teacher_logits, mask, self.temperature)


class CosineProbs(DistilLoss):
    """在概率空间计算Cosine distance（推荐）"""
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return cosine_probs(logits, teacher_logits, mask, self.temperature)

