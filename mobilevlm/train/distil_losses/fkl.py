"""
This implementation is based on [DistiLLM's](https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L4)
"""
from typing import Optional
import torch
from torch.nn import functional as F
from .base import DistilLoss


def forward_kl(
    logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    teacher_probs: Optional[torch.Tensor] = None,
    student_logprobs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # 确保所有张量的序列长度一致
    seq_lens = [logits.shape[1], teacher_logits.shape[1], mask.shape[1]]
    if teacher_probs is not None:
        seq_lens.append(teacher_probs.shape[1])
    if student_logprobs is not None:
        seq_lens.append(student_logprobs.shape[1])
    min_seq_len = min(seq_lens)
    
    logits = logits[:, :min_seq_len, :]
    teacher_logits = teacher_logits[:, :min_seq_len, :]
    mask = mask[:, :min_seq_len]
    
    if teacher_probs is None:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    else:
        teacher_probs = teacher_probs[:, :min_seq_len, :]
    
    if student_logprobs is None:
        student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    else:
        student_logprobs = student_logprobs[:, :min_seq_len, :]
    
    inf_mask = torch.isinf(logits)
    prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


class ForwardKL(DistilLoss):
    def forward(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return forward_kl(logits, teacher_logits, mask)
