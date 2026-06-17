"""
Visual-Odor cross-modal contrastive learning model.

This module combines vision and smell encoders for learning aligned
representations across visual and olfactory modalities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

# Import vision backbones
from models.vision_encoder.clip import CLIPVisionBackbone
from models.vision_encoder.dino import DINOv3VisionBackbone

# Import smell backbone
from models.smell_encoder.transformer import Transformer

# Import aligner
from models.aligner import LinearAligner, ResidualMLPAligner

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
import os

# Only print from main process (rank 0) in distributed training
rank = int(os.environ.get('RANK', -1))
console = Console(quiet=(rank > 0))


class ModelConfig:
    """Configuration for VisualOdorModel"""
    
    def __init__(
        self,
        # Vision settings
        vision_backbone: str = "dino",
        vision_model_name: str = "dinov3_vitb16",
        vision_clip_pretrained: str = "datacomp_xl_s13b_b90k",  # For CLIP
        vision_dino_repo_local: str = "ext_repos/dinov3",
        vision_use_pretrained: bool = True,
        vision_forward_option: str = "spatial_tokens", # 'cls_token' or 'spatial_tokens'
        vision_projection_type: str = "aligner",
        
        # Smell settings
        smell_input_dim: int = 12,
        smell_model_dim: int = 256,
        smell_num_heads: int = 8,
        smell_num_layers: int = 4,
        smell_use_cls_token: bool = True,
        smell_forward_option: str = "cls_token", # 'cls_token' or 'spatial_tokens'
        smell_use_positional_encoding: bool = True,
        smell_projection_type: str = "aligner", # 'aligner', 'residual_mlp', or 'none'
        
        # Projection settings
        target_embedding_dim: int = 768,
        use_norm: bool = True,
        
        # Training settings
        init_logit_scale: float = np.log(1 / 0.07),
        dropout: float = 0.1,
        
        # Freeze settings
        vision_freeze_backbone: bool = False,
        vision_freeze_projection: bool = False,
        smell_freeze_backbone: bool = False,
        smell_freeze_projection: bool = False,
    ):
        # Vision
        self.vision_backbone = vision_backbone
        self.vision_model_name = vision_model_name
        self.vision_clip_pretrained = vision_clip_pretrained
        self.vision_dino_repo_local = vision_dino_repo_local
        self.vision_use_pretrained = vision_use_pretrained
        self.vision_forward_option = vision_forward_option
        self.vision_projection_type = vision_projection_type
        
        # Smell
        self.smell_input_dim = smell_input_dim
        self.smell_model_dim = smell_model_dim
        self.smell_num_heads = smell_num_heads
        self.smell_num_layers = smell_num_layers
        self.smell_use_cls_token = smell_use_cls_token
        self.smell_forward_option = smell_forward_option
        self.smell_use_positional_encoding = smell_use_positional_encoding
        self.smell_projection_type = smell_projection_type
        
        # Projection
        self.target_embedding_dim = target_embedding_dim
        self.use_norm = use_norm
        
        # Training
        self.init_logit_scale = init_logit_scale
        self.dropout = dropout
        
        # Freeze
        self.vision_freeze_backbone = vision_freeze_backbone
        self.vision_freeze_projection = vision_freeze_projection
        self.smell_freeze_backbone = smell_freeze_backbone
        self.smell_freeze_projection = smell_freeze_projection
        
        # Validation
        self._validate_config()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.vision_backbone in ['clip', 'dino'], \
            f"vision_backbone must be 'clip' or 'dino', got {self.vision_backbone}"
        
        assert self.vision_forward_option in ['cls_token', 'spatial_tokens'], \
            f"vision_forward_option must be 'cls_token' or 'spatial_tokens', got {self.vision_forward_option}"
        
        assert self.vision_projection_type in ['aligner', 'none'], \
            f"vision_projection_type must be 'aligner' or 'none', got {self.vision_projection_type}"
        
        assert self.smell_projection_type in ['aligner', 'residual_mlp', 'none'], \
            f"smell_projection_type must be 'aligner', 'residual_mlp', or 'none', got {self.smell_projection_type}"
        
        assert self.smell_forward_option in ['cls_token', 'spatial_tokens'], \
            f"smell_forward_option must be 'cls_token' or 'spatial_tokens', got {self.smell_forward_option}"


class BaseEncoder(nn.Module):
    """Base class for modality encoders"""
    
    def __init__(self, config: ModelConfig, modality_name: str):
        super().__init__()
        self.config = config
        self.modality_name = modality_name
        self.backbone = None
        self.projection = None
    
    def _setup_backbone(self):
        """Setup backbone encoder - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _setup_backbone")
    
    def _setup_projection(self):
        """Setup projection layer - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _setup_projection")
    
    def _apply_projection(self, features: torch.Tensor) -> torch.Tensor:
        """Apply projection if available"""
        raise NotImplementedError("Subclasses must implement _apply_projection")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward")
    
    def freeze_backbone(self):
        """Freeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = False
            console.print(f"[dim]→ {self.modality_name} backbone frozen[/dim]")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        if self.backbone is not None:
            for param in self.backbone.parameters():
                param.requires_grad = True
            console.print(f"[dim]→ {self.modality_name} backbone unfrozen[/dim]")
    
    def freeze_projection(self):
        """Freeze projection/aligner parameters"""
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = False
            console.print(f"[dim]→ {self.modality_name} projection frozen[/dim]")
    
    def unfreeze_projection(self):
        """Unfreeze projection/aligner parameters"""
        if self.projection is not None:
            for param in self.projection.parameters():
                param.requires_grad = True
            console.print(f"[dim]→ {self.modality_name} projection unfrozen[/dim]")
    
    def set_freeze_status(self, freeze_backbone: bool, freeze_projection: bool):
        """Set freeze status for backbone and projection"""
        if freeze_backbone:
            self.freeze_backbone()
        else:
            self.unfreeze_backbone()
        
        if freeze_projection:
            self.freeze_projection()
        else:
            self.unfreeze_projection()


class VisionEncoder(BaseEncoder):
    """Vision modality encoder"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, 'vision')
        self._setup_backbone()
        self._setup_projection()
        
        # Apply freeze settings
        self.set_freeze_status(
            freeze_backbone=config.vision_freeze_backbone,
            freeze_projection=config.vision_freeze_projection
        )
        
        vision_info = Table(show_header=False, box=None, padding=(0, 1))
        vision_info.add_column(style="dim")
        vision_info.add_column()
        vision_info.add_row("Backbone", f"[cyan]{config.vision_backbone}[/cyan]")
        vision_info.add_row("Forward option", f"[cyan]{config.vision_forward_option}[/cyan]")
        vision_info.add_row("Projection", f"[cyan]{config.vision_projection_type}[/cyan]")
        vision_info.add_row("Backbone frozen", f"[cyan]{config.vision_freeze_backbone}[/cyan]")
        vision_info.add_row("Projection frozen", f"[cyan]{config.vision_freeze_projection}[/cyan]")
        
        console.print(Panel(
            vision_info,
            title="[bold cyan]VisionEncoder Initialized[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
    
    def _setup_backbone(self):
        """Setup vision backbone (CLIP or DINO)"""
        if self.config.vision_backbone == 'clip':
            self.backbone = CLIPVisionBackbone(
                model_name=self.config.vision_model_name,
                pretrained=self.config.vision_clip_pretrained,
                forward_option=self.config.vision_forward_option
            )
            self.feature_dim = self.backbone.get_feature_dim()
        
        elif self.config.vision_backbone == 'dino':
            self.backbone = DINOv3VisionBackbone(
                model_name=self.config.vision_model_name,
                use_pretrained=self.config.vision_use_pretrained,
                repo_local=self.config.vision_dino_repo_local,
                forward_option=self.config.vision_forward_option
            )
            self.feature_dim = self.backbone.get_feature_dim()
    
    def _setup_projection(self):
        """Setup projection layer (aligner)"""
        if self.config.vision_projection_type == 'aligner':
            self.projection = LinearAligner(
                in_dim=self.feature_dim,
                out_dim=self.config.target_embedding_dim,
                use_norm=self.config.use_norm,
            )
            console.print(f"  [dim]Vision aligner: {self.feature_dim} → {self.config.target_embedding_dim}[/dim]")
        
        elif self.config.vision_projection_type == 'none':
            self.projection = None
            console.print(f"  [dim]No vision projection layer[/dim]")
    
    def _apply_projection(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to features.
        
        Args:
            features: [B, D] or [B, D, H, W]
        
        Returns:
            projected: [B, D'] or [B, D', H, W]
        """
        if self.projection is None:
            return features
        
        # Handle different input shapes
        if features.dim() == 2:
            # CLS token: [B, D] → use cls_layer
            _, cls_out = self.projection(spatial=None, cls=features)
            return cls_out  # [B, D']
        
        elif features.dim() == 4:
            # Spatial tokens: [B, D, H, W] → use spatial layer
            spatial_out, _ = self.projection(spatial=features, cls=None)
            return spatial_out  # [B, D', H, W]
        
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for vision encoder.
        
        Args:
            image: [B, 3, H, W] input images
        
        Returns:
            features: [B, D] or [B, D, H, W] normalized features
        """
        # Extract features from backbone
        features = self.backbone(image)
        
        # Apply projection
        features = self._apply_projection(features)
        
        # Normalize along channel dimension
        if features.dim() == 2:
            # [B, D]
            features = F.normalize(features, dim=1)
        elif features.dim() == 4:
            # [B, D, H, W]
            features = F.normalize(features, dim=1)
        
        return features


class SmellEncoder(BaseEncoder):
    """Smell modality encoder"""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config, 'smell')
        self._setup_backbone()
        self._setup_projection()
        
        # Apply freeze settings
        self.set_freeze_status(
            freeze_backbone=config.smell_freeze_backbone,
            freeze_projection=config.smell_freeze_projection
        )
        
        smell_info = Table(show_header=False, box=None, padding=(0, 1))
        smell_info.add_column(style="dim")
        smell_info.add_column()
        smell_info.add_row("Model dim", f"[cyan]{config.smell_model_dim}[/cyan]")
        smell_info.add_row("Num layers", f"[cyan]{config.smell_num_layers}[/cyan]")
        smell_info.add_row("Pool", f"[cyan]{config.smell_forward_option}[/cyan]")
        smell_info.add_row("Projection", f"[cyan]{config.smell_projection_type}[/cyan]")
        smell_info.add_row("Backbone frozen", f"[cyan]{config.smell_freeze_backbone}[/cyan]")
        smell_info.add_row("Projection frozen", f"[cyan]{config.smell_freeze_projection}[/cyan]")
        
        console.print(Panel(
            smell_info,
            title="[bold cyan]SmellEncoder Initialized[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
    
    def _setup_backbone(self):
        """Setup smell backbone (Transformer)"""
        self.backbone = Transformer(
            input_dim=self.config.smell_input_dim,
            model_dim=self.config.smell_model_dim,
            num_classes=0,  # Not used for contrastive learning
            num_heads=self.config.smell_num_heads,
            num_layers=self.config.smell_num_layers,
            dropout=self.config.dropout,
            use_positional_encoding=self.config.smell_use_positional_encoding,
            use_cls_token=self.config.smell_use_cls_token,
            pool=self.config.smell_forward_option
        )
        self.feature_dim = self.config.smell_model_dim
    
    def _setup_projection(self):
        """Setup projection layer (aligner)"""
        if self.config.smell_projection_type == 'aligner':
            # Baseline: single linear projection
            self.projection = LinearAligner(
                in_dim=self.feature_dim,
                out_dim=self.config.target_embedding_dim,
                use_norm=self.config.use_norm,
            )
            console.print(f"  [dim]Smell LinearAligner: {self.feature_dim} → {self.config.target_embedding_dim}[/dim]")

        elif self.config.smell_projection_type == 'residual_mlp':
            # Recommended: two-layer MLP with residual connection
            # Norm → Linear → GELU → Norm → Linear + residual
            self.projection = ResidualMLPAligner(
                in_dim=self.feature_dim,
                out_dim=self.config.target_embedding_dim,
            )
            console.print(f"  [dim]Smell ResidualMLPAligner: {self.feature_dim} → {self.config.target_embedding_dim}[/dim]")

        elif self.config.smell_projection_type == 'none':
            self.projection = None
            console.print(f"  [dim]No smell projection layer[/dim]")
    
    def _apply_projection(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply projection to features.

        Args:
            features: [B, D] for 'aligner' / 'residual_mlp' / 'none'

        Returns:
            projected: [B, D']
        """
        if self.projection is None:
            return features

        if self.config.smell_projection_type == 'residual_mlp':
            # ResidualMLPAligner operates directly on [B, D]
            return self.projection(features)  # [B, D']

        else:
            # LinearAligner / TwoLayerAligner: use cls path
            _, cls_out = self.projection(spatial=None, cls=features)
            return cls_out  # [B, D']
    
    def forward(self, sensor_data: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for smell encoder.
        
        Args:
            sensor_data: [B, T, F] sensor sequences
            lengths: [B] actual sequence lengths (optional)
        
        Returns:
            features: [B, D] normalized features
        """
        # Extract features from backbone (no classifier)
        features = self.backbone.forward_tokens(sensor_data, lengths) # [B, T, D]
        
        # Apply projection
        features = self._apply_projection(features)
        
        # mean over time
        features = features.mean(dim=1) if features.dim() == 3 else features  # [B, D]

        # Normalize
        features = F.normalize(features, dim=1)
        
        return features


class VisualOdorModel(nn.Module):
    """
    Visual-Odor cross-modal contrastive learning model.
    
    Combines vision and smell encoders for learning aligned representations.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Create encoders
        self.vision_encoder = VisionEncoder(config)
        self.smell_encoder = SmellEncoder(config)
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(
            torch.ones([]) * config.init_logit_scale
        )
        
        model_info = Table(show_header=False, box=None, padding=(0, 1))
        model_info.add_column(style="dim")
        model_info.add_column()
        model_info.add_row("Target embedding dim", f"[cyan]{config.target_embedding_dim}[/cyan]")
        model_info.add_row("Initial logit scale", f"[cyan]{config.init_logit_scale:.4f}[/cyan]")
        
        console.print(Panel(
            model_info,
            title="[bold cyan]VisualOdorModel Initialized[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
    
    def forward(
        self, 
        vision: Optional[torch.Tensor] = None,
        smell: Optional[torch.Tensor] = None,
        smell_lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            vision: [B, 3, H, W] images (optional)
            smell: [B, T, F] sensor data (optional)
            smell_lengths: [B] actual lengths (optional)
        
        Returns:
            dict with keys:
                - 'vision': [B, D] or [B, D, H, W] vision features (if vision provided)
                - 'smell': [B, D] smell features (if smell provided)
                - 'logit_scale': scalar temperature parameter
        """
        output = {}
        
        if vision is not None:
            output['vision'] = self.vision_encoder(vision)
        
        if smell is not None:
            output['smell'] = self.smell_encoder(smell, smell_lengths)
        
        # Clamp logit_scale to prevent explosion (max exp(4.6) = 100)
        with torch.no_grad():
            self.logit_scale.clamp_(max=4.6)
        
        output['logit_scale'] = self.logit_scale.exp()
        
        return output
    
    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract vision features only"""
        return self.vision_encoder(images)
    
    def get_smell_features(
        self, 
        sensor_data: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Extract smell features only"""
        return self.smell_encoder(sensor_data, lengths)
