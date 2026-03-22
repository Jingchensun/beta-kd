import torch
import torch.nn.functional as F
try:
    from lightning import LightningModule
except ImportError:
    LightningModule = None
from .base import DistilLoss
from .fkl import forward_kl


class TAID(DistilLoss):
    def __init__(
        self,
        t_start: float = 0.4,
        t_end: float = 1.0,
        alpha: float = 5e-4,
        beta: float = 0.99,
        disable_adaptive: bool = False,
    ):
        super().__init__()
        # validation
        assert 0.0 <= t_start < 1.0
        assert 0.0 < t_end <= 1.0
        assert 0.0 <= alpha <= 1.0

        self.t_start = t_start
        self.t_end = t_end
        self.alpha = alpha
        self.beta = beta
        self.disable_adaptive = disable_adaptive
        self.register_buffer(
            "t", torch.tensor(t_start, device="cuda", dtype=torch.float32)
        )
        self.register_buffer(
            "prev_loss", torch.tensor(float("inf"), device="cuda", dtype=torch.float32)
        )
        self.register_buffer(
            "momentum", torch.zeros([], device="cuda", dtype=torch.float32)
        )

    def update_t(
        self, loss: torch.Tensor, global_step: int, num_train_steps: int
    ) -> torch.Tensor:
        if torch.isinf(self.prev_loss):
            self.prev_loss = loss
            return
        # Calculate relative change rate
        relative_change = (self.prev_loss - loss) / (self.prev_loss + 1e-15)
        # Update momentum
        self.momentum = self.beta * self.momentum + (1 - self.beta) * relative_change

        # Calculate adaptive delta
        adaptive_delta = torch.sigmoid(self.momentum)
        # Update t (ensure monotonic increase)
        progress = global_step / num_train_steps
        t_target = self.t_start + (self.t_end - self.t_start) * progress
        delta_t = self.alpha * adaptive_delta * (1 - self.t)
        t = (
            min(self.t_end, max(t_target, self.t + delta_t))
            if not self.disable_adaptive
            else t_target
        )
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.t.device, dtype=self.t.dtype)
        self.t = t
        self.prev_loss = loss
        return delta_t

    def compute_loss(
        self,
        logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        mask: torch.Tensor,
    ):
        # 先对齐序列长度，避免维度不匹配
        min_seq_len = min(logits.shape[1], teacher_logits.shape[1], mask.shape[1])
        logits_aligned = logits[:, :min_seq_len, :]
        teacher_logits_aligned = teacher_logits[:, :min_seq_len, :]
        mask_aligned = mask[:, :min_seq_len]
        
        # 使用对齐后的张量计算p_t
        p_t = (1 - self.t) * logits_aligned.detach() + self.t * teacher_logits_aligned
        p_t = F.softmax(p_t, dim=-1, dtype=torch.float32)
        distil_loss = forward_kl(
            logits=logits_aligned,
            teacher_logits=teacher_logits_aligned,
            mask=mask_aligned,
            teacher_probs=p_t,
        )
        return distil_loss

    def forward(
        self,
        lightning_module=None,
        logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
        mask: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        # compute kd loss
        loss = self.compute_loss(logits, teacher_logits, mask)

        # update t
        trainer = getattr(lightning_module, 'trainer', type('', (), {'global_step': 0, 'estimated_stepping_batches': 1000})())
        delta_t = self.update_t(
            loss.detach().clone(),
            global_step=trainer.global_step,
            num_train_steps=trainer.estimated_stepping_batches,
        )

        loss_dict = {
            "distil_loss": loss,
            "tiki_t": self.t,
            "delta_t": delta_t,
        }
        return loss_dict
