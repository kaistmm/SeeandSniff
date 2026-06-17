"""
Aligner module for projecting encoder features to shared embedding space.

This module provides projection layers that align features from different modalities
into a common latent space for contrastive learning.

Modules
-------
LinearAligner       : Single linear projection (baseline)
TwoLayerAligner     : Two-layer MLP projection
ResidualMLPAligner  : Two-layer MLP with residual connection (recommended upgrade)
"""

import torch
import torch.nn as nn


class Identity2(nn.Module):
    """Identity module that passes through two inputs unchanged."""
    
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x, y


class ChannelNorm(nn.Module):
    """Channel-wise Layer Normalization for spatial features."""
    
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = nn.LayerNorm(dim, eps=1e-4)

    def forward_spatial(self, x):
        """Normalize spatial features [B, C, H, W]"""
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

    def forward(self, x, cls):
        """Normalize both spatial and cls features"""
        return self.forward_spatial(x), self.forward_cls(cls)

    def forward_cls(self, cls):
        """Normalize cls token features"""
        if cls is not None:
            return self.norm(cls)
        else:
            return None

def id_conv(dim, strength=0.9):
    """
    Create an identity-initialized 1x1 convolution.
    
    Args:
        dim: Number of channels
        strength: Strength of identity initialization (0-1)
    
    Returns:
        nn.Conv2d initialized close to identity mapping
    """
    conv = nn.Conv2d(dim, dim, 1, padding="same")
    
    with torch.no_grad():
        identity = torch.eye(dim, device=conv.weight.device, dtype=conv.weight.dtype)
        identity = identity.reshape(dim, dim, 1, 1)
        
        new_weight = identity * strength + conv.weight * (1 - strength)
        new_bias = conv.bias * (1 - strength)
        
        conv.weight = nn.Parameter(new_weight.contiguous())
        conv.bias = nn.Parameter(new_bias.contiguous())
    
    return conv




class LinearAligner(nn.Module):
    """
    Linear aligner for projecting features to target dimension.
    
    Features:
    - Channel-wise normalization
    - 1x1 convolution for spatial features
    - Linear projection for cls tokens
    - Optional self-attention for feature refinement
    - Optional attention pooling for feature aggregation
    """
    
    def __init__(
        self, 
        in_dim, 
        out_dim, 
        use_norm=True,
    ):
        super().__init__()

        # Normalization
        self.norm = ChannelNorm(in_dim) if use_norm else Identity2()
        
        # Spatial projection (1x1 conv)
        if in_dim == out_dim:
            self.layer = id_conv(in_dim, 0)  # Identity with strength=0
        else:
            self.layer = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)
        
        # CLS token projection
        self.cls_layer = nn.Linear(in_dim, out_dim)


    def forward(self, spatial=None, cls=None):
        """
        Args:
            spatial: [B, C, H, W] spatial features (optional)
            cls: [B, D] cls token features (optional)
        
        Returns:
            spatial_out: [B, out_dim, H', W'] projected spatial features (if spatial provided)
            cls_out: [B, out_dim] projected cls features (if cls provided)
        """
        # Handle cls-only case
        if spatial is None and cls is not None:
            norm_cls = self.norm.forward_cls(cls)
            aligned_cls = self.cls_layer(norm_cls)
            return None, aligned_cls
        
        # Handle spatial-only or both cases
        norm_spatial, norm_cls = self.norm(spatial, cls)
        
        # Project cls token if provided
        aligned_cls = self.cls_layer(norm_cls) if norm_cls is not None else None
        
        # Project spatial features
        processed_spatial = self.layer(norm_spatial)

        return processed_spatial, aligned_cls


class TwoLayerAligner(nn.Module):
    """
    Two-layer aligner for projecting features to target dimension.
    
    Features:
    - Channel-wise normalization
    - Two 1x1 convolutions with ReLU activation for spatial features
    - Two-layer MLP for cls tokens
    - Hidden dimension for intermediate representation
    """
    
    def __init__(
        self, 
        in_dim, 
        out_dim,
        hidden_dim=None,
        use_norm=True,
    ):
        super().__init__()
        
        # Default hidden dimension is the average of in_dim and out_dim
        if hidden_dim is None:
            hidden_dim = (in_dim + out_dim) // 2

        # Normalization
        self.norm = ChannelNorm(in_dim) if use_norm else Identity2()
        
        # Spatial projection (two 1x1 convs with activation)
        self.layer1 = nn.Conv2d(in_dim, hidden_dim, kernel_size=1, stride=1)
        self.activation = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1)
        
        # CLS token projection (two-layer MLP)
        self.cls_layer1 = nn.Linear(in_dim, hidden_dim)
        self.cls_layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, spatial=None, cls=None):
        """
        Args:
            spatial: [B, C, H, W] spatial features (optional)
            cls: [B, D] cls token features (optional)
        
        Returns:
            spatial_out: [B, out_dim, H', W'] projected spatial features (if spatial provided)
            cls_out: [B, out_dim] projected cls features (if cls provided)
        """
        # Handle cls-only case
        if spatial is None and cls is not None:
            norm_cls = self.norm.forward_cls(cls)
            hidden_cls = self.activation(self.cls_layer1(norm_cls))
            aligned_cls = self.cls_layer2(hidden_cls)
            return None, aligned_cls
        
        # Handle spatial-only or both cases
        norm_spatial, norm_cls = self.norm(spatial, cls)
        
        # Project cls token if provided
        if norm_cls is not None:
            hidden_cls = self.activation(self.cls_layer1(norm_cls))
            aligned_cls = self.cls_layer2(hidden_cls)
        else:
            aligned_cls = None
        
        # Project spatial features through two layers
        hidden_spatial = self.activation(self.layer1(norm_spatial))
        processed_spatial = self.layer2(hidden_spatial)

        return processed_spatial, aligned_cls


class ResidualMLPAligner(nn.Module):
    """
    Two-layer MLP aligner with residual connection.

    Architecture (for [B, D] smell features):
        x → LayerNorm → Linear(in, hidden) → GELU → LayerNorm → Linear(hidden, out)
                                                                         │
        (residual Linear if in_dim != out_dim)  ─────────────────────────┘

    Compared to TwoLayerAligner, the residual path stabilizes training when
    in_dim ≈ out_dim and helps preserve information through the projection.
    Recommended as a drop-in upgrade over LinearAligner for smell features.

    Args:
        in_dim:     Input feature dimension
        out_dim:    Output feature dimension
        hidden_dim: Intermediate dimension (default: max(in_dim, out_dim))
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)
            # hidden_dim = 192

        self.norm1 = nn.LayerNorm(in_dim, eps=1e-4)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-4)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # Residual path: project if dimensions differ, else identity
        self.residual = (
            nn.Linear(in_dim, out_dim, bias=False)
            if in_dim != out_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] pooled smell features
        Returns:
            [B, out_dim] projected features
        """
        residual = self.residual(x)
        out = self.fc2(self.norm2(self.act(self.fc1(self.norm1(x)))))
        return out + residual
