"""
MSE-based distillation losses for knowledge distillation.

提供两种MSE实现：
1. MSE_Logits: 在logits空间计算MSE（需要归一化处理）
2. MSE_Probs: 在概率空间计算MSE（推荐，更稳定）
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def mse_logits(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    在logits空间计算MSE，先进行标准化（standardization）再计算MSE。
    
    改进：参考cosine distance的思路，先对logits进行标准化（零均值、单位方差），
    消除均值和方差的影响，只关注分布形状。
    
    Args:
        logits: 学生模型的logits [batch, seq_len, vocab_size]
        teacher_logits: 教师模型的logits [batch, seq_len, vocab_size]
        mask: 有效位置的mask [batch, seq_len]
        temperature: 温度参数，用于缩放logits（默认1.0）
    
    Returns:
        MSE损失值（标量）
    """
    # 确保所有张量的序列长度一致
    min_seq_len = min(logits.shape[1], teacher_logits.shape[1], mask.shape[1])
    logits = logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    mask = mask[:, :min_seq_len]
    
    # 检查nan和inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("[MSE-Logits] Warning: Student logits contain NaN or Inf values")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        print("[MSE-Logits] Warning: Teacher logits contain NaN or Inf values")
        return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    
    # 应用温度缩放（软化分布）
    if temperature != 1.0:
        logits = logits / temperature
        teacher_logits = teacher_logits / temperature
    
    # 【标准化】对logits进行标准化：(x - mean) / std，消除均值和方差的影响
    # 参考cosine distance的思路，只关注分布形状
    logits_mean = logits.mean(dim=-1, keepdim=True)
    logits_std = logits.std(dim=-1, keepdim=True) + 1e-8  # 避免除零
    logits_normalized = (logits - logits_mean) / logits_std
    
    teacher_logits_mean = teacher_logits.mean(dim=-1, keepdim=True)
    teacher_logits_std = teacher_logits.std(dim=-1, keepdim=True) + 1e-8
    teacher_logits_normalized = (teacher_logits - teacher_logits_mean) / teacher_logits_std
    
    # 计算逐位置的MSE [batch, seq_len, vocab_size] -> [batch, seq_len]
    mse_per_position = ((logits_normalized - teacher_logits_normalized) ** 2).mean(dim=-1)
    
    # 使用mask加权平均
    mask_sum = mask.sum()
    if mask_sum == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    mse_loss = (mse_per_position * mask).sum() / mask_sum
    
    # 如果使用了温度缩放，需要调整损失尺度（类似标准KD）
    if temperature != 1.0:
        mse_loss = mse_loss * (temperature ** 2)
    
    # 检查最终结果是否为nan/inf
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        print(f"[MSE-Logits] Warning: Final MSE loss is NaN or Inf")
        print(f"  - logits_normalized range: [{logits_normalized.min():.4f}, {logits_normalized.max():.4f}]")
        print(f"  - teacher_logits_normalized range: [{teacher_logits_normalized.min():.4f}, {teacher_logits_normalized.max():.4f}]")
        print(f"  - mse_per_position range: [{mse_per_position.min():.4f}, {mse_per_position.max():.4f}]")
        print(f"  - mask_sum: {mask_sum}")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    return mse_loss


def mse_probs(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    在概率空间计算MSE（推荐方法）。
    
    将logits转换为概率分布后计算MSE，更稳定且有界。
    这种方法关注预测分布的差异，而不是原始logits值。
    
    Args:
        logits: 学生模型的logits [batch, seq_len, vocab_size]
        teacher_logits: 教师模型的logits [batch, seq_len, vocab_size]
        mask: 有效位置的mask [batch, seq_len]
        temperature: 温度参数，用于软化概率分布（默认1.0）
    
    Returns:
        MSE损失值（标量）
    """
    # 确保所有张量的序列长度一致
    min_seq_len = min(logits.shape[1], teacher_logits.shape[1], mask.shape[1])
    logits = logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    mask = mask[:, :min_seq_len]
    
    # 检查nan和inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("[MSE-Probs] Warning: Student logits contain NaN or Inf values")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
        print("[MSE-Probs] Warning: Teacher logits contain NaN or Inf values")
        return torch.tensor(0.0, device=teacher_logits.device, dtype=teacher_logits.dtype)
    
    # 转换为概率分布（使用温度缩放）
    student_probs = F.softmax(logits / temperature, dim=-1, dtype=torch.float32)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
    
    # 处理inf值（将其概率设为0）
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)
    student_probs = torch.masked_fill(student_probs, inf_mask, 0.0)
    teacher_probs = torch.masked_fill(teacher_probs, inf_mask, 0.0)
    
    # 计算逐位置的MSE [batch, seq_len, vocab_size] -> [batch, seq_len]
    mse_per_position = ((student_probs - teacher_probs) ** 2).mean(dim=-1)
    
    # 使用mask加权平均
    mask_sum = mask.sum()
    if mask_sum == 0:
        print("[MSE-Probs] Warning: mask_sum is 0, no valid positions for distillation")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    mse_loss = (mse_per_position * mask).sum() / mask_sum
    
    # 调试信息：打印损失值和相关统计
    print(f"[MSE-Probs DEBUG] mse_loss: {mse_loss.item():.6f}, mask_sum: {mask_sum.item():.1f}, "
          f"mse_per_position range: [{mse_per_position.min().item():.6f}, {mse_per_position.max().item():.6f}]")
    
    # 如果使用了温度缩放，可以选择调整损失尺度
    # 注意：在概率空间，温度缩放的影响已经体现在概率分布中
    # 通常不需要像KD那样乘以temperature^2，但可以根据实验调整
    if temperature != 1.0:
        mse_loss = mse_loss * (temperature ** 2)
    
    # 检查最终结果是否为nan/inf
    if torch.isnan(mse_loss) or torch.isinf(mse_loss):
        print(f"[MSE-Probs] Warning: Final MSE loss is NaN or Inf")
        print(f"  - student_probs range: [{student_probs.min():.4f}, {student_probs.max():.4f}]")
        print(f"  - teacher_probs range: [{teacher_probs.min():.4f}, {teacher_probs.max():.4f}]")
        print(f"  - mse_per_position range: [{mse_per_position.min():.4f}, {mse_per_position.max():.4f}]")
        print(f"  - mask_sum: {mask_sum}")
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    return mse_loss


class MSE_Logits(DistilLoss):
    """在logits空间计算MSE"""
    
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
        return mse_logits(logits, teacher_logits, mask, self.temperature)


class MSE_Probs(DistilLoss):
    """在概率空间计算MSE（推荐）"""
    
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
        return mse_probs(logits, teacher_logits, mask, self.temperature)


# 默认使用概率空间的MSE
class MSE(MSE_Probs):
    """MSE蒸馏损失（默认在概率空间计算）"""
    pass

