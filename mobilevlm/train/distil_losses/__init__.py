from .base import DistilLoss
from .fkl import ForwardKL
from .rkl import ReverseKL
from .tvd import TVD
from .js import JS
from .adaptive_kl import AdaptiveKL
from .skew_fkl import SkewForwardKL
from .skew_rkl import SkewReverseKL
from .ctkd import CTKD
from .ctkd_mlp import CTKDMLP, CTKD_MLP
from .dkd import DKD
from .taid import TAID
from .mse import MSE, MSE_Logits, MSE_Probs
from .cosine import Cosine, CosineProbs
