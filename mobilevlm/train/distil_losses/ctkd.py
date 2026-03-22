"""
Curriculum Temperature for Knowledge Distillation
https://arxiv.org/abs/2211.16231

This implementation is based on https://github.com/zhengli97/CTKD
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from lightning import LightningModule
except ImportError:
    LightningModule = None
from .base import DistilLoss


# copied from https://github.com/zhengli97/CTKD/blob/master/models/temp_global.py#L21
class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = lambda_ * grads
        # print(dx)
        return dx, None


class GradientReversal(torch.nn.Module):
    def __init__(self):
        super(GradientReversal, self).__init__()
        # self.lambda_ = lambda_

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class CTKD(DistilLoss):
    """
    Implementation of CTKD for Language Modeling
    """

    def __init__(
        self,
        lambda_max: float = 1,
        lambda_min: float = 0,
        num_loops: int = 10,
        temp_start: float = 1,
        temp_end: float = 4,  # 降低温度上限
    ):
        super().__init__()
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min
        self.num_loops = num_loops
        self.temp_start = temp_start
        self.temp_end = temp_end
        # 初始化温度参数为较小的值，避免初始温度过高
        self.global_temperature = nn.Parameter(torch.tensor(-2.0))  # 负值使sigmoid输出较小
        # In their experiments, Global-T is used as default
        self.grl = GradientReversal()

    def get_value(self, epoch):
        if epoch < 0:
            epoch = 0
        if epoch >= self.num_loops:
            epoch = self.num_loops
        value = (math.cos(epoch * math.pi / self.num_loops) + 1.0) * 0.5
        value = value * (self.lambda_max - self.lambda_min) + self.lambda_min
        return value

    def forward(
        self,
        lightning_module=None,
        logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
        mask: torch.Tensor = None,
        batch=None,
        **kwargs,
    ) -> torch.Tensor:
        epoch = getattr(lightning_module, 'trainer', type('', (), {'current_epoch': 0})()).current_epoch + 1
        lambda_ = self.get_value(epoch)
        temp = self.grl(self.global_temperature, lambda_)
        # 限制温度范围，使用更合理的范围
        temp = torch.clamp(self.temp_start + self.temp_end * torch.sigmoid(temp), min=0.5, max=4.0)
        
        # 确保logits和teacher_logits的形状一致
        if logits.shape != teacher_logits.shape:
            min_seq_len = min(logits.shape[1], teacher_logits.shape[1])
            logits = logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # 确保mask与logits的前两维形状一致
        if mask is not None and mask.shape[:2] != logits.shape[:2]:
            min_seq_len = min(mask.shape[1], logits.shape[1])
            mask = mask[:, :min_seq_len]
            logits = logits[:, :min_seq_len, :]
            teacher_logits = teacher_logits[:, :min_seq_len, :]
        
        # 数值稳定性：限制logits范围
        logits = torch.clamp(logits, min=-100, max=100)
        teacher_logits = torch.clamp(teacher_logits, min=-100, max=100)
        
        # forward kl
        teacher_probs = F.softmax(teacher_logits / temp, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits / temp, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        
        # 计算每个token的损失
        x = torch.sum(prod_probs, dim=-1)  # shape: (batch_size, seq_len)
        
        # 确保mask与x的形状一致
        if mask is not None:
            # 展平并应用mask
            x_flat = x.reshape(-1)  # shape: (batch_size * seq_len,)
            mask_flat = mask.reshape(-1)  # shape: (batch_size * seq_len,)
            
            # 计算加权损失
            masked_x = x_flat * mask_flat
            valid_tokens = mask_flat.sum()
            
            if valid_tokens > 0:
                distil_loss = -masked_x.sum() / valid_tokens
            else:
                distil_loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            # 如果没有mask，直接计算平均损失
            distil_loss = -x.mean()
        
        # 使用更温和的温度缩放
        distil_loss = distil_loss * temp  # 只乘以一次温度，而不是平方
        
        # 添加调试信息（每100步打印一次）
        if hasattr(lightning_module, '_step_counter'):
            lightning_module._step_counter = getattr(lightning_module, '_step_counter', 0) + 1
        else:
            lightning_module._step_counter = 1
            
        if lightning_module._step_counter % 100 == 0:
            print(f"[CTKD Debug] Epoch: {epoch}, Lambda: {lambda_:.4f}, Temp: {temp.item():.4f}, Loss: {distil_loss.item():.4f}")
        
        return distil_loss
