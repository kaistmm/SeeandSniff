"""
CLIP-based vision encoder.

Provides a simple wrapper around OpenCLIP models for extracting visual features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from typing import Optional


class CLIPVisionBackbone(nn.Module):
    """
    CLIP vision encoder using OpenCLIP.
    
    Extracts visual features from images using pretrained CLIP models.
    Supports different pooling strategies (CLS token or average pooling).
    """
    
    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "datacomp_xl_s13b_b90k",
        forward_option: str = "cls_token"
    ):
        """
        Args:
            model_name: CLIP model architecture (e.g., 'ViT-L-14', 'ViT-B-16')
            pretrained: Pretrained weights identifier
            forward_option: Feature extraction mode ('cls_token' or 'spatial_tokens')
        """
        super().__init__()
        
        assert forward_option in ['cls_token', 'spatial_tokens'], \
            f"forward_option must be 'cls_token' or 'spatial_tokens', got {forward_option}"
        
        self.forward_option = forward_option
        self.model_name = model_name
        self.pretrained = pretrained
        
        # Load CLIP model
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        
        self.clip = clip_model
        self.preprocess = preprocess
        
        # Get feature dimension
        self.cls_token_dim = self.clip.visual.output_dim  # encode_image dim
        
        # Get spatial token dimension (hidden dim of ViT)
        if hasattr(self.clip.visual, 'width'):
            self.spatial_token_dim = self.clip.visual.width
        elif hasattr(self.clip.visual, 'embed_dim'):
            self.spatial_token_dim = self.clip.visual.embed_dim
        elif hasattr(self.clip.visual, 'trunk') and hasattr(self.clip.visual.trunk, 'embed_dim'):
            self.spatial_token_dim = self.clip.visual.trunk.embed_dim
        elif hasattr(self.clip.visual, 'conv1'):
            # Try to infer from conv1 output channels
            self.spatial_token_dim = self.clip.visual.conv1.out_channels
        else:
            # Fallback to output_dim if we can't find hidden dim
            self.spatial_token_dim = self.cls_token_dim
        
        self.feature_dim = self.cls_token_dim if forward_option == "cls_token" else self.spatial_token_dim
        
        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.clip = self.clip.to(self.device)
        
        # Freeze by default (typically used as frozen feature extractor)
        for param in self.clip.visual.parameters():
            param.requires_grad = False
    
    def get_feature_dim(self):
        """Return feature dimension based on forward option"""
        return self.feature_dim
    
    def forward_spatial(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial patch features from CLIP vision transformer.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            spatial_features: [B, D, H, W] spatial patch features
        """
        image = image.to(self.device)
        visual = self.clip.visual
        
        # Patch projection
        x = visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # [B, H*W, D_model]
        
        # Add CLS token
        cls_token = visual.class_embedding.to(x.dtype) + \
                    torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)  # [B, 1+H*W, D_model]
        
        # Add positional embedding (with interpolation for different resolutions)
        pos_embed = visual.positional_embedding.to(x.dtype)  # [1+N_pretrain, D]
        if pos_embed.shape[0] != x.shape[1]:
            # Separate CLS pos embed and spatial pos embeds, then interpolate spatial
            cls_pos = pos_embed[:1]                               # [1, D]
            spatial_pos = pos_embed[1:]                           # [N_pretrain, D]
            N_pretrain = spatial_pos.shape[0]
            H_pretrain = W_pretrain = int(N_pretrain ** 0.5)
            N_new = x.shape[1] - 1                               # number of new spatial tokens
            H_new = W_new = int(N_new ** 0.5)
            # Interpolate: [1, D, H_pre, W_pre] → [1, D, H_new, W_new]
            spatial_pos = spatial_pos.reshape(1, H_pretrain, W_pretrain, -1).permute(0, 3, 1, 2).float()
            spatial_pos = F.interpolate(spatial_pos, size=(H_new, W_new), mode='bicubic', align_corners=False)
            spatial_pos = spatial_pos.permute(0, 2, 3, 1).reshape(N_new, -1).to(x.dtype)
            pos_embed = torch.cat([cls_pos, spatial_pos], dim=0)  # [1+N_new, D]
        x = x + pos_embed
        
        # Pass through transformer blocks
        x = x.permute(1, 0, 2)  # [1+H*W, B, D_model]
        for layer in visual.transformer.resblocks:
            x = layer(x)
        x = x.permute(1, 0, 2)  # [B, 1+H*W, D_model]
        
        # Apply final layer norm
        x = visual.ln_post(x)  # [B, 1+H*W, D_model]
        
        # Extract spatial features (exclude CLS token)
        spatial_features = x[:, 1:, :]  # [B, H*W, D_model]
        
        # # Apply CLIP's projection layer if exists (BUT it's cls token specific projection)
        if hasattr(visual, 'proj') and visual.proj is not None:
            # Project to output dimension: [B, H*W, D_model] @ [D_model, D_out]
            spatial_features = spatial_features @ visual.proj  # [B, H*W, D_out]
        
        # Reshape to spatial format [B, D, H, W]
        B, N, D = spatial_features.shape
        H = W = int(N**0.5)
        if H * W != N:
            raise ValueError(f"Number of patches ({N}) is not a perfect square")
        
        spatial_features = spatial_features.reshape(B, H, W, D).permute(0,3,1,2)
        
        return spatial_features
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            features: [B, D] if cls_token mode
                     [B, D, H', W'] if spatial_tokens mode 
        """
        if self.forward_option == "cls_token":
            # Use CLIP's default forward (CLS token)
            with torch.no_grad():
                image = image.to(self.device)
                features = self.clip.encode_image(image, normalize=True)  # [B, D]
            return features  # [B, D]
        
        elif self.forward_option == "spatial_tokens":
            # Extract spatial patch features
            with torch.no_grad():
                spatial_features = self.forward_spatial(image)  # [B, D, H, W]
            
            # !!! Need to project to common feature dim if used in alignment && normalization is needed
            
            return spatial_features
        
        else:
            raise ValueError(f"Unknown forward_option: {self.forward_option}")
    
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim
