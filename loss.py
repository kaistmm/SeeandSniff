"""
Visual-Odor Contrastive Loss

Cross-modal contrastive learning loss between vision and smell modalities.
Supports both CLS token and spatial token modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import json

from rich.console import Console
from rich.panel import Panel
from rich import box
import os

# Only print from main process (rank 0) in distributed training
rank = int(os.environ.get('RANK', -1))
console = Console(quiet=(rank > 0))


class VisualOdorLoss(nn.Module):
    """
    Contrastive loss for Visual-Odor cross-modal learning.
    
    Supports two modes:
    - CLS mode: vision [B, D], smell [B, D] → standard CLIP loss
    - Spatial mode: vision [B, D, H, W], smell [B, D] → aggregation loss
    
    Args:
        forward_mode: 'cls' or 'spatial' - determines which loss computation to use
        spatial_pool: Pooling method for spatial aggregation ('max' or 'mean')
        retrieval_mode: 'samplewise' or 'ingredientwise' retrieval evaluation
        label_json_path: Path to ingredient_labels.json (for ingredientwise mode)
    """
    
    def __init__(
        self,
        forward_mode: str = 'global',
        spatial_pool: str = 'mean',
        retrieval_mode: str = 'ingredientwise',
        label_json_path: str = 'metadata/ingredient_labels.json',
    ):
        super().__init__()
        assert forward_mode in ['global', 'local'], f"forward_mode must be 'global' or 'local', got {forward_mode}"
        self.forward_mode = forward_mode
        self.spatial_pool = spatial_pool
        self.retrieval_mode = retrieval_mode
        
        # Load ingredient metadata (for ingredientwise retrieval)
        if retrieval_mode == 'ingredientwise':
            if os.path.exists(label_json_path):
                with open(label_json_path, 'r') as f:
                    self.label_metadata = json.load(f)
            else:
                console.print(f"[yellow]⚠ Label file not found: {label_json_path}[/yellow]")
                self.label_metadata = None
        else:
            self.label_metadata = None
        
        info_lines = [f"[cyan]Forward mode:[/cyan] {self.forward_mode}"]
        if self.forward_mode == 'local':
            info_lines.append(f"[cyan]Spatial pooling:[/cyan] {self.spatial_pool}")
        info_lines.append(f"[cyan]Retrieval mode:[/cyan] {self.retrieval_mode}")
        
        console.print(Panel(
            "\n".join(info_lines),
            title="[bold cyan]VisualOdorLoss Initialized[/bold cyan]",
            border_style="cyan",
            box=box.ROUNDED
        ))
    
    def contrastive_loss_cls(
        self, 
        vision_feat: torch.Tensor, 
        smell_feat: torch.Tensor, 
        logit_scale: torch.Tensor
    ):
        """
        Standard CLIP-style contrastive loss for CLS tokens.
        
        Args:
            vision_feat: [B, D] vision features
            smell_feat: [B, D] smell features
            logit_scale: scalar temperature parameter
        
        Returns:
            loss: scalar loss value
            affinity_matrix: [B, B] similarity matrix
        """
        B = vision_feat.shape[0]
        labels = torch.arange(B, device=vision_feat.device, dtype=torch.long)
        
        # Flatten if needed
        vision_feat = vision_feat.reshape(B, -1)
        smell_feat = smell_feat.reshape(B, -1)
        
        # Compute similarity matrix
        affinity_matrix = logit_scale * vision_feat @ smell_feat.T  # [B, B]
        
        # Symmetric cross-entropy loss
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        
        loss = (row_loss + col_loss) / 2
        
        return loss, affinity_matrix
    
    def compute_samplewise_retrieval(
        self,
        affinity_matrix: torch.Tensor,
        topk=(1, 5, 10)
    ):
        """
        Sample-wise retrieval accuracy (standard CLIP-style).
        
        Assumes diagonal elements are positive pairs.
        
        Args:
            affinity_matrix: [B, B] similarity matrix
            topk: tuple of k values for top-k accuracy
        
        Returns:
            dict with accuracy metrics
        """
        B = affinity_matrix.shape[0]
        device = affinity_matrix.device
        labels = torch.arange(B, device=device, dtype=torch.long)
        
        # Limit topk to batch size
        valid_topk = tuple(k for k in topk if k <= B)
        if not valid_topk:
            valid_topk = (min(1, B),)
        
        results = {}
        
        # Vision → Smell retrieval
        topk_indices = affinity_matrix.topk(max(valid_topk), dim=1).indices  # [B, max_k]
        for k in valid_topk:
            topk_k = topk_indices[:, :k]  # [B, k]
            # Check if ground truth label is in top-k
            correct = (topk_k == labels.unsqueeze(1)).any(dim=1).float()
            acc = correct.mean() * 100
            results[f'v2s_acc{k}'] = acc
        
        # Smell → Vision retrieval (transpose)
        topk_indices_t = affinity_matrix.T.topk(max(valid_topk), dim=1).indices
        for k in valid_topk:
            topk_k = topk_indices_t[:, :k]
            correct = (topk_k == labels.unsqueeze(1)).any(dim=1).float()
            acc = correct.mean() * 100
            results[f's2v_acc{k}'] = acc
        
        return results
    
    def compute_ingredientwise_retrieval(
        self,
        affinity_matrix: torch.Tensor,
        query_labels: torch.Tensor,
        retrieved_labels: torch.Tensor,
        topk=(1, 5, 10)
    ):
        """
        Ingredient-wise retrieval accuracy.
        
        Samples with the same ingredient label are considered correct matches.
        
        Args:
            affinity_matrix: [N_query, N_retrieved] similarity matrix
                            rows = query samples, columns = retrieved samples
            query_labels: [N_query] ingredient indices for query samples
            retrieved_labels: [N_retrieved] ingredient indices for retrieved samples
            topk: tuple of k values for top-k accuracy
        
        Returns:
            dict with accuracy metrics (v2s_acc and s2v_acc)
        """
        N_query = affinity_matrix.shape[0]
        N_retrieved = affinity_matrix.shape[1]
        device = affinity_matrix.device
        query_labels = query_labels.to(device)
        retrieved_labels = retrieved_labels.to(device)
        
        # Limit topk to retrieved set size
        valid_topk = tuple(k for k in topk if k <= N_retrieved)
        if not valid_topk:
            valid_topk = (min(1, N_retrieved),)
        
        results = {}
        
        # Vision → Smell retrieval (row: vision query, column: smell retrieved)
        topk_indices = affinity_matrix.topk(max(valid_topk), dim=1).indices  # [N_query, max_k]
        
        for k in valid_topk:
            topk_k = topk_indices[:, :k]  # [N_query, k]
            retrieved_k = retrieved_labels[topk_k]  # [N_query, k]
            query_k = query_labels.unsqueeze(1).expand_as(retrieved_k)  # [N_query, k]
            
            correct = (retrieved_k == query_k).any(dim=1).float()
            acc = correct.mean() * 100
            results[f'v2s_acc{k}'] = acc
        
        # Smell → Vision retrieval (transpose: row=smell query, column=vision retrieved)
        valid_topk_t = tuple(k for k in topk if k <= N_query)
        if not valid_topk_t:
            valid_topk_t = (min(1, N_query),)
            
        topk_indices_t = affinity_matrix.T.topk(max(valid_topk_t), dim=1).indices  # [N_retrieved, max_k]
        
        for k in valid_topk_t:
            topk_k = topk_indices_t[:, :k]  # [N_retrieved, k]
            retrieved_k = query_labels[topk_k]  # [N_retrieved, k]
            query_k = retrieved_labels.unsqueeze(1).expand_as(retrieved_k)  # [N_retrieved, k]
            
            correct = (retrieved_k == query_k).any(dim=1).float()
            acc = correct.mean() * 100
            results[f's2v_acc{k}'] = acc
        
        return results
    
    def contrastive_loss_spatial(
        self,
        vision_feat: torch.Tensor,
        smell_feat: torch.Tensor,
        logit_scale: torch.Tensor,
        pool: str = 'mean',
    ):
        """
        Spatial aggregation contrastive loss.
        
        Computes similarity heatmap between smell and vision spatial tokens,
        then aggregates via pooling.
        
        Args:
            vision_feat: [B, D, H, W] spatial vision features
            smell_feat: [B, D] smell features
            logit_scale: scalar temperature parameter
            pool: 'max' or 'mean' pooling
        
        Returns:
            loss: scalar loss value
            affinity_matrix: [B, B] aggregated similarity matrix
        """
        B, D, H, W = vision_feat.shape
        labels = torch.arange(B, device=vision_feat.device, dtype=torch.long)
        
        # Reshape vision: [B, H*W, D]
        vision_feat = vision_feat.reshape(B, D, H*W).permute(0, 2, 1)  # [B, H*W, D]
        
        # Expand smell: [B, 1, D]
        smell_feat = smell_feat.unsqueeze(1)  # [B, 1, D]
        
        # Compute similarity heatmap: [B, B, H*W]
        sim_heatmap = torch.einsum("bnd,kmd->bkn", vision_feat, smell_feat)
        sim_heatmap = sim_heatmap * logit_scale
        
        # Aggregate similarity heatmap
        if pool == 'mean':
            affinity_matrix = sim_heatmap.mean(dim=-1)  # [B, B]
        elif pool == 'max':
            affinity_matrix = sim_heatmap.max(dim=-1)[0]  # [B, B]
        else:
            raise ValueError(f"Unsupported pooling method: {pool}")
        
        # Symmetric cross-entropy loss
        row_loss = F.cross_entropy(affinity_matrix, labels)
        col_loss = F.cross_entropy(affinity_matrix.T, labels)
        
        loss = (row_loss + col_loss) / 2
        
        return loss, affinity_matrix
    
    def forward(
        self,
        output_dict: dict,
        ingredient_labels: torch.Tensor = None,
        return_metrics: bool = False
    ):
        """
        Compute contrastive loss.
        
        Args:
            output_dict: {
                'vision': [B, D] or [B, D, H, W],
                'smell': [B, D],
                'logit_scale': scalar
            }
            ingredient_labels: [B] ingredient indices (required for ingredientwise mode)
            return_metrics: If True, return dict with loss and metrics
        
        Returns:
            If return_metrics=False: scalar loss
            If return_metrics=True: dict with 'loss' and metrics
        """
        vision_feat = output_dict['vision']
        smell_feat = output_dict['smell']
        logit_scale = output_dict['logit_scale']
        
        # Apply loss based on forward_mode
        if self.forward_mode == 'global':
            # CLS mode: [B, D]
            loss, affinity_matrix = self.contrastive_loss_cls(
                vision_feat, smell_feat, logit_scale
            )
        elif self.forward_mode == 'local':
            # Spatial mode: [B, D, H, W]
            loss, affinity_matrix = self.contrastive_loss_spatial(
                vision_feat, smell_feat, logit_scale, pool=self.spatial_pool
            )
        else:
            raise ValueError(f"Unsupported forward_mode: {self.forward_mode}")
        
        if return_metrics:
            metrics = {
                'loss': loss,
                'affinity_matrix': affinity_matrix,
                'logit_scale': logit_scale.item()
            }
            
            # Compute retrieval accuracy based on mode
            if self.retrieval_mode == 'samplewise':
                retrieval_metrics = self.compute_samplewise_retrieval(
                    affinity_matrix,
                    topk=(1, 5, 10)
                )
                metrics.update(retrieval_metrics)
            
            elif self.retrieval_mode == 'ingredientwise':
                if ingredient_labels is None:
                    raise ValueError("ingredient_labels required for ingredientwise retrieval")
                retrieval_metrics = self.compute_ingredientwise_retrieval(
                    affinity_matrix,
                    ingredient_labels,
                    topk=(1, 5, 10)
                )
                metrics.update(retrieval_metrics)
            
            return metrics
        else:
            return loss
