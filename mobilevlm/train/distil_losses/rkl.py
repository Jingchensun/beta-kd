"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L14)
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def reverse_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    student_probs: Optional[torch.Tensor] = None,
    teacher_logprobs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 确保logits和teacher_logits形状匹配
    if logits.shape != teacher_logits.shape:
        # 如果形状不匹配，截取较短的长度
        min_seq_len = min(logits.shape[1], teacher_logits.shape[1])
        logits = logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
    
    # 确保mask形状与logits的前两维匹配
    if mask.shape != logits.shape[:2]:
        min_seq_len = min(mask.shape[1], logits.shape[1])
        mask = mask[:, :min_seq_len]
        logits = logits[:, :min_seq_len, :]
        teacher_logits = teacher_logits[:, :min_seq_len, :]
    
    if student_probs is None:
        student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    if teacher_logprobs is None:
        teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    inf_mask = torch.isinf(teacher_logits) | torch.isinf(logits)
    prod_probs = torch.masked_fill(student_probs * teacher_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    
    # 确保x和mask展平后的形状匹配
    mask_flat = mask.view(-1)
    if x.shape[0] != mask_flat.shape[0]:
        min_len = min(x.shape[0], mask_flat.shape[0])
        x = x[:min_len]
        mask_flat = mask_flat[:min_len]
    
    # 避免除零错误
    mask_sum = torch.sum(mask_flat, dim=0)
    if mask_sum == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    
    distil_loss = -torch.sum(x * mask_flat, dim=0) / mask_sum
    return distil_loss


class ReverseKL(DistilLoss):
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return reverse_kl(logits, teacher_logits, mask)
