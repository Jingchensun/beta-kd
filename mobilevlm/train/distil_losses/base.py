from typing import Union, Dict, Any
import torch
from torch import nn

try:
    from lightning.pytorch import LightningModule
except ImportError:
    # 如果lightning不可用，创建一个dummy类
    LightningModule = Any


class DistilLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lightning_module=None,
        logits: torch.Tensor = None,
        teacher_logits: torch.Tensor = None,
        mask: torch.Tensor = None,
        batch: Dict = None,
        **kwargs,
    ) -> Union[Dict, torch.Tensor]:
        raise NotImplementedError
