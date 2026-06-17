import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Iterable, List, Sequence, Union


# ---------- GCMS MLP Encoder ----------
class GCMSMLPEncoder(nn.Module):
    """
    GCMS encoder for flat vectors.
    Input:  x (B, N)  # N = number of GCMS points per sample
    Output: z (B, embedding_dim)
    """
    def __init__(
        self,
        in_features: int,                 # N
        embedding_dim: int = 256,
        hidden: tuple[int, ...] = (512, 256),
        dropout: float = 0.1,
        use_layernorm: bool = True,       # per-sample feature normalization at input
        use_batchnorm: bool = False,      # BN between hidden layers (off by default for small batches)
        l2_normalize: bool = False,       # set True if your contrastive loss expects unit vectors
    ):
        super().__init__()
        layers: list[nn.Module] = []

        if use_layernorm:
            layers.append(nn.LayerNorm(in_features))

        last = in_features
        for h in hidden:
            layers.append(nn.Linear(last, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            last = h

        layers.append(nn.Linear(last, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.l2_normalize = l2_normalize

    def forward_features(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() != 2:
            # Allow accidental (B, T, C) by flattening, but prefer giving (B, N)
            x = x.view(x.size(0), -1)
        z = self.net(x)  # (B, D)
        if self.l2_normalize:
            z = F.normalize(z, dim=-1)
        return z

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_features(x, lengths)