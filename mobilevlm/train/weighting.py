"""
Loss weighting strategies for knowledge distillation.

Supported strategies:
1. type1 / equal    (EqualWeighting): simple sum with equal weights
2. type2 / task     (HeteroscedasticUncertainty): task-level uncertainty weighting
3. type3 / instance (InstanceConditionalWeighting): instance-conditional weighting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Type 1: Equal Weighting (baseline)
# ============================================================================
class EqualWeighting(nn.Module):
    """Simple sum of all losses with equal weights."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: variable number of scalar losses
            teacher_features: unused, kept for interface consistency
        Returns:
            total_loss: scalar
            weights: dict for logging
        """
        total_loss = sum(losses)
        
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        for i, loss in enumerate(losses):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = 1.0
            weights[f'{name}_weighted'] = loss.item() if hasattr(loss, 'item') else loss
        
        return total_loss, weights


# ============================================================================
# Type 2: Heteroscedastic Uncertainty (Kendall et al. 2018)
# ============================================================================
class HeteroscedasticUncertainty(nn.Module):
    """
    Task uncertainty-based loss weighting (Kendall et al. 2018).

    Formula: L_total = Σ_i (1 / (2σ_i²)) * L_i + log(σ_i)
    Only requires learning num_tasks scalar parameters.
    """
    
    def __init__(self, num_tasks=2):
        super().__init__()
        self.num_tasks = num_tasks
        # Initialized to 0.0 so that sigma = exp(0.0) = 1.0, precision = 1.0.
        # Using float32 to avoid fp16 gradient clipping.
        # For faster convergence, consider uniform(-1, 1) initialization.
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float32))
        
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: variable number of scalar losses
            teacher_features: unused, kept for interface consistency
        Returns:
            total_loss: weighted total loss
            weights: dict for logging
        """
        # Clamp log_vars to [-5, 5] to avoid numerical overflow.
        # This keeps sigma in [e^-5, e^5] ≈ [0.0067, 148.4].
        # self.log_vars is already float32, no need to cast (avoids breaking grad).
        log_vars_clamped = torch.clamp(self.log_vars, min=-5.0, max=5.0)
        precision = torch.exp(-log_vars_clamped)
        
        total_loss = 0
        for i, loss in enumerate(losses):
            # Cast to float32 for numerically stable computation
            loss_fp32 = loss.float() if loss.dtype == torch.float16 else loss
            weighted = 0.5 * precision[i] * loss_fp32 + 0.5 * log_vars_clamped[i]
            # Cast back to original dtype
            total_loss += weighted.to(loss.dtype) if loss.dtype == torch.float16 else weighted
        
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        for i, loss in enumerate(losses):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = (0.5 * precision[i]).item()
            loss_fp32 = loss.float() if loss.dtype == torch.float16 else loss
            weights[f'{name}_weighted'] = (0.5 * precision[i] * loss_fp32).item()
        
        return total_loss, weights


# ============================================================================
# Type 3: Instance-Conditional Weighting
# ============================================================================
class InstanceConditionalWeighting(nn.Module):
    """
    Instance-conditional loss weighting.
    An MLP predicts per-batch task weights from teacher visual features.
    """
    
    def __init__(self, feature_dim=4096, num_tasks=2, hidden_dim=128):
        super().__init__()
        self.num_tasks = num_tasks
        self.uncertainty_predictor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tasks)
        )
    
    def forward(self, *losses, teacher_features=None):
        """
        Args:
            *losses: variable number of scalar losses
            teacher_features: (batch_size, feature_dim) or (batch_size, seq_len, feature_dim)
        Returns:
            total_loss: weighted total loss
            weights: dict for logging
        """
        if teacher_features is None:
            # Fallback to equal weighting
            return EqualWeighting().forward(*losses)
        
        # Average over sequence dimension if needed
        if teacher_features.dim() == 3:
            teacher_features = teacher_features.mean(dim=1)
        
        batch_size = teacher_features.shape[0]
        
        # Predict log_vars and clamp to avoid numerical overflow
        log_vars = self.uncertainty_predictor(teacher_features.detach())
        log_vars = torch.clamp(log_vars, min=-5.0, max=5.0)
        
        # Expand scalar losses to batch dimension
        losses_tensor = []
        for loss in losses:
            loss_batch = loss.unsqueeze(0).repeat(batch_size) if loss.dim() == 0 else loss
            losses_tensor.append(loss_batch)
        losses_tensor = torch.stack(losses_tensor, dim=1)
        
        precision = torch.exp(-log_vars)
        weighted_losses = precision * losses_tensor + log_vars
        total_loss = torch.mean(torch.sum(weighted_losses, dim=1))
        
        task_names = ['main_loss', 'logits_distill_loss', 'v_loss', 'attn_loss']
        weights = {}
        individual_weights = torch.mean(precision, dim=0)
        individual_weighted_losses = torch.mean(weighted_losses, dim=0)
        
        for i in range(len(losses)):
            name = task_names[i] if i < len(task_names) else f'task_{i}'
            weights[f'{name}_weight'] = individual_weights[i].item()
            weights[f'{name}_weighted'] = individual_weighted_losses[i].item()
        
        return total_loss, weights


# ============================================================================
# Factory function
# ============================================================================
def create_weighting_strategy(strategy_type, **kwargs):
    """
    Create a loss weighting strategy.

    Args:
        strategy_type: one of:
            - "type1" or "equal"    : equal weighting
            - "type2" or "task"     : heteroscedastic uncertainty (task-level)
            - "type3" or "instance" : instance-conditional weighting
        **kwargs: passed to the strategy constructor

    Returns:
        WeightingStrategy module
    """
    strategy_map = {
        'type1':    EqualWeighting,
        'equal':    EqualWeighting,
        'type2':    HeteroscedasticUncertainty,
        'task':     HeteroscedasticUncertainty,
        'type3':    InstanceConditionalWeighting,
        'instance': InstanceConditionalWeighting,
    }
    
    strategy_type = strategy_type.lower()
    if strategy_type not in strategy_map:
        print(f"Unknown strategy type: {strategy_type}, falling back to type1 (equal)")
        strategy_type = 'type1'
    
    strategy_class = strategy_map[strategy_type]
    
    if strategy_class == EqualWeighting:
        return strategy_class()
    elif strategy_class == HeteroscedasticUncertainty:
        num_tasks = kwargs.get('num_tasks', 2)
        return strategy_class(num_tasks=num_tasks)
    else:  # Type 3
        feature_dim = kwargs.get('feature_dim', 4096)
        num_tasks = kwargs.get('num_tasks', 2)
        hidden_dim = kwargs.get('hidden_dim', 128)
        return strategy_class(feature_dim=feature_dim, num_tasks=num_tasks, hidden_dim=hidden_dim)
