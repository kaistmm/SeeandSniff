"""
DINOv3-based vision encoder.

Provides a wrapper around DINOv3 models for extracting visual features.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DINOv3VisionBackbone(nn.Module):
    """
    DINOv3 vision encoder.
    
    Extracts visual features from images using pretrained DINOv3 models.
    Supports both CLS token and spatial patch token extraction.
    """
    
    def __init__(
        self,
        model_name: str = "dinov3_vits16",
        use_pretrained: bool = True,
        repo_local: str = "ext_repos/dinov3",
        forward_option: str = "cls_token"
    ):
        """
        Args:
            model_name: DINOv3 model variant ('dinov3_vits16' or 'dinov3_vitb16')
            use_pretrained: Whether to load pretrained weights
            repo_local: Local path to DINOv3 repository
            forward_option: Feature extraction mode ('cls_token' or 'spatial_tokens')
        """
        super().__init__()
        
        assert forward_option in ['cls_token', 'spatial_tokens'], \
            f"forward_option must be 'cls_token' or 'spatial_tokens', got {forward_option}"
        
        assert model_name in ['dinov3_vits16', 'dinov3_vitb16'], \
            f"model_name must be 'dinov3_vits16' or 'dinov3_vitb16', got {model_name}"
        
        self.model_name = model_name
        self.use_pretrained = use_pretrained
        self.forward_option = forward_option
        
        # Validate local repository
        if not os.path.isdir(repo_local):
            raise FileNotFoundError(
                f"DINOv3 local repo not found: {repo_local}\n"
                f"→ git clone https://github.com/facebookresearch/dinov3 {repo_local}"
            )
        
        REPO_DIR = os.path.abspath(repo_local)
        
        # Load model based on variant
        if model_name == 'dinov3_vits16':
            self.feature_dim = 384
            weights_path = os.path.join(
                repo_local, 
                'ckpt_dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'
            ) if use_pretrained else None
            
            self.model = torch.hub.load(
                REPO_DIR,
                'dinov3_vits16',
                source='local',
                weights=weights_path
            )
            
        elif model_name == 'dinov3_vitb16':
            self.feature_dim = 768
            weights_path = os.path.join(
                repo_local,
                'ckpt_dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
            ) if use_pretrained else None
            
            self.model = torch.hub.load(
                REPO_DIR,
                'dinov3_vitb16',
                source='local',
                weights=weights_path
            )
        
        # Move to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        # Freeze by default
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward_cls_token(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract CLS token features.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            cls_features: [B, D] CLS token features
        """
        image = image.to(self.device)
        # Use model's default forward (returns CLS token)
        features = self.model(image)  # [B, D]
        return features
    
    def forward_spatial_tokens(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial patch token features.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            spatial_features: [B, H, W, D] spatial patch features
        """
        image = image.to(self.device)
        # Use forward_features to get all tokens
        output = self.model.forward_features(image)
        patch_tokens = output["x_norm_patchtokens"]  # [B, N, D]
        
        # Reshape to spatial format
        B, N, D = patch_tokens.shape
        H = W = int(N ** 0.5)
        
        if H * W != N:
            raise ValueError(
                f"Number of patches ({N}) is not a perfect square. "
                f"Cannot reshape to [B, D, H, W]"
            )
        
        # Reshape: [B, N, D] → [B, D, H, W]
        spatial_features = patch_tokens.reshape(B, H, W, D).permute(0, 3, 1, 2)
        
        return spatial_features
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features from images.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            features: [B, D] if cls_token mode
                     [B, D, H, W] if spatial_tokens mode
        """
        with torch.no_grad():
            if self.forward_option == "cls_token":
                return self.forward_cls_token(image)
            elif self.forward_option == "spatial_tokens":
                return self.forward_spatial_tokens(image)
            else:
                raise ValueError(f"Unknown forward_option: {self.forward_option}")
    
    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.feature_dim
