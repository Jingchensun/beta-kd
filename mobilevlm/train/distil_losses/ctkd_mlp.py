"""
Curriculum Temperature for Knowledge Distillation with MLP Temperature Learning
基于MLP网络学习温度系数的CTKD实现

主要特点：
1. 使用两层MLP网络学习温度系数
2. 温度系数同时应用于teacher和student的logits
3. 返回KD loss用于与主任务loss相加
4. 共同优化MLP网络和student网络参数
"""

import torch
from torch import nn
from torch.nn import functional as F
from .base import DistilLoss


class CTKDMLP(DistilLoss):
    """
    CTKD with MLP-based temperature learning
    
    该实现使用MLP网络来学习温度系数，而不是使用梯度反转层。
    MLP网络接收student和teacher的logits作为输入，输出温度系数。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        temp_min: float = 0.5,
        temp_max: float = 4.0,
        vocab_size: int = 32000,  # 默认词汇表大小
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # 两层MLP网络用于学习温度系数
        self.temperature_mlp = nn.Sequential(
            nn.Linear(vocab_size * 2, hidden_dim),  # 输入：student + teacher logits
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),  # 输出：温度系数
            nn.Sigmoid()  # 确保输出在[0,1]范围内
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化MLP权重"""
        for module in self.temperature_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _compute_temperature(self, student_logits, teacher_logits):
        """
        使用MLP网络计算温度系数
        
        Args:
            student_logits: 学生模型logits, shape: (batch_size, seq_len, vocab_size)
            teacher_logits: 教师模型logits, shape: (batch_size, seq_len, vocab_size)
            
        Returns:
            temperature: 学习到的温度系数, shape: (batch_size, seq_len, 1)
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # 将logits展平为(batch_size * seq_len, vocab_size)
        student_flat = student_logits.view(-1, vocab_size)
        teacher_flat = teacher_logits.view(-1, vocab_size)
        
        # 拼接student和teacher的logits作为MLP输入
        combined_input = torch.cat([student_flat, teacher_flat], dim=1)
        
        # 通过MLP网络
        temp_raw = self.temperature_mlp(combined_input)  # shape: (batch_size * seq_len, 1)
        
        # 将温度映射到指定范围
        temperature = self.temp_min + (self.temp_max - self.temp_min) * temp_raw
        
        # 重塑回原始形状
        temperature = temperature.view(batch_size, seq_len, 1)
        
        return temperature
    
    def forward(
        self,
        lightning_module=None,
        logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
        mask: torch.Tensor = None,
        batch=None,
        **kwargs,
    ) -> torch.Tensor:
        """
        计算CTKD-MLP损失
        
        Args:
            lightning_module: 训练模块（用于调试信息）
            logits: 学生模型logits
            teacher_logits: 教师模型logits
            mask: 掩码，用于忽略padding tokens
            batch: 批次数据
            **kwargs: 其他参数
            
        Returns:
            distil_loss: 知识蒸馏损失
        """
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
        
        # 使用MLP学习温度系数
        temperature = self._compute_temperature(logits, teacher_logits)
        
        # 使用学习到的温度计算概率分布
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1, dtype=torch.float32)
        student_logprobs = F.log_softmax(logits / temperature, dim=-1, dtype=torch.float32)
        
        # 处理无穷大值
        inf_mask = torch.isinf(logits)
        prod_probs = torch.masked_fill(teacher_probs * student_logprobs, inf_mask, 0)
        
        # 计算每个token的损失
        x = torch.sum(prod_probs, dim=-1)  # shape: (batch_size, seq_len)
        
        # 应用mask计算加权损失
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
        
        # 使用温度缩放损失（可选，根据具体需求调整）
        # distil_loss = distil_loss * temperature.mean()
        
        # 添加调试信息（每100步打印一次）
        if hasattr(lightning_module, '_step_counter'):
            lightning_module._step_counter = getattr(lightning_module, '_step_counter', 0) + 1
        else:
            lightning_module._step_counter = 1
            
        if lightning_module._step_counter % 100 == 0:
            avg_temp = temperature.mean().item()
            print(f"[CTKD-MLP Debug] Temp: {avg_temp:.4f}, Loss: {distil_loss.item():.4f}")
        
        return distil_loss


# 为了向后兼容，保留CTKDMLP的别名
CTKD_MLP = CTKDMLP 