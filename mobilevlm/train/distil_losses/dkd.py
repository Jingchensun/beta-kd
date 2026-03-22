"""
Decoupled Knowledge Distillation
https://arxiv.org/abs/2203.08679

This implementation is based on https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
"""
import torch
from torch.nn import functional as F
from .base import DistilLoss


def normalize(logit):
    mean = logit.mean(dim=-1, keepdims=True)
    stdv = logit.std(dim=-1, keepdims=True)
    return (logit - mean) / (1e-7 + stdv)


def dkd_loss(
    logits_student_in,
    logits_teacher_in,
    target,
    mask,
    alpha=1.0,
    beta=5,
    temperature=1.0,
    logit_stand=False,
):
    # 对齐序列长度：取最小长度以确保维度匹配
    batch_size = logits_student_in.size(0)
    min_seq_len = min(logits_student_in.size(1), target.size(1))
    
    # 裁剪到相同的序列长度
    logits_student_in = logits_student_in[:, :min_seq_len, :]
    logits_teacher_in = logits_teacher_in[:, :min_seq_len, :]
    target = target[:, :min_seq_len]
    mask = mask[:, :min_seq_len]
    
    logits_student_in = logits_student_in.reshape(-1, logits_student_in.size(-1))
    logits_teacher_in = logits_teacher_in.reshape(-1, logits_student_in.size(-1))
    target = target.flatten()
    mask = mask.flatten()
    
    # 确保target在有效范围内
    vocab_size = logits_student_in.size(-1)
    target = torch.clamp(target, min=0, max=vocab_size - 1)

    logits_student = normalize(logits_student_in) if logit_stand else logits_student_in
    logits_teacher = normalize(logits_teacher_in) if logit_stand else logits_teacher_in

    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    
    # 添加数值稳定性：避免log(0)
    pred_student = torch.clamp(pred_student, min=1e-8)
    pred_teacher = torch.clamp(pred_teacher, min=1e-8)
    log_pred_student = torch.log(pred_student)
    
    tckd_loss = F.kl_div(log_pred_student, pred_teacher, reduction="none")
    tckd_loss = tckd_loss * mask.reshape(-1).unsqueeze(1)
    
    # 避免除以0：使用mask的有效数量
    num_valid = torch.sum(mask) + 1e-8
    tckd_loss = torch.sum(tckd_loss) * (temperature**2) / num_valid

    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none")
    nckd_loss = nckd_loss * mask.reshape(-1).unsqueeze(1)
    nckd_loss = torch.sum(nckd_loss) * (temperature**2) / num_valid
    
    total_loss = alpha * tckd_loss + beta * nckd_loss
    
    # 检查并处理NaN
    if torch.isnan(total_loss):
        print(f"[DKD Warning] NaN detected! tckd_loss: {tckd_loss.item()}, nckd_loss: {nckd_loss.item()}, num_valid: {num_valid.item()}")
        total_loss = torch.zeros_like(total_loss)
    
    return total_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt


class DKD(DistilLoss):
    """
    Implementation of DKD for Language Modeling
    paper: https://arxiv.org/abs/2203.08679
    code: https://github.com/megvii-research/mdistiller/blob/master/mdistiller/distillers/DKD.py
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.5, temperature: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(
        self,
        lightning_module=None,
        logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
        mask: torch.Tensor = None,
        batch=None,
        **kwargs,
    ) -> torch.Tensor:
        labels = batch.get("model_inputs", {}).get("labels", torch.zeros_like(logits[:, :, 0])).clone()
        
        if labels.dim() > 1:
            labels = labels[..., 1:]
            # 对logits、teacher_logits和mask进行相应的切片以匹配labels的维度
            logits = logits[:, :-1, :]
            teacher_logits = teacher_logits[:, :-1, :]
            if mask is not None and mask.dim() > 1:
                mask = mask[:, :-1]
        
        tokenizer = getattr(lightning_module, 'tokenizer', type('', (), {'pad_token_id': 0})())
        labels[labels == (-100)] = tokenizer.pad_token_id

        return dkd_loss(
            logits_student_in=logits,
            logits_teacher_in=teacher_logits,
            target=labels,
            mask=mask,
            alpha=self.alpha,
            beta=self.beta,
            temperature=self.temperature,
        )
