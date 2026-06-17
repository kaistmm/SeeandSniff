"""
Linear Probing for Smell Encoder.

This script loads a pretrained Visual-Odor model checkpoint,
freezes the smell encoder, and trains a linear classifier on top
for ingredient classification.

Usage:
    python linear_probing_smell.py --config configs/SeeandSniff.yaml --checkpoint path/to/checkpoint.pth
"""

import argparse
import json
import yaml
import numpy as np
import os
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model import VisualOdorModel, ModelConfig
from load_data import (
    load_sensor_data,
    create_label_encoder_from_json,
    diff_data_like,
    make_sliding_window_dataset,
)
from dataset import group_windows_by_ingredient

console = Console()


class SmellClassificationDataset(Dataset):
    """
    Dataset for smell-based classification.
    
    Args:
        smell_data: dict[ingredient, list[Tensor(T,F)]] - same format as main_pretrain
        le: LabelEncoder for ingredient labels
    """
    
    def __init__(self, smell_data, le):
        self.smell_data = smell_data
        self.le = le
        
        # Build index: [(ingredient, sample_idx, label_idx), ...]
        self.samples = []
        for ingredient in sorted(smell_data.keys()):
            label_idx = le.transform([ingredient])[0]
            for sample_idx in range(len(smell_data[ingredient])):
                self.samples.append((ingredient, sample_idx, label_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        ingredient, sample_idx, label_idx = self.samples[idx]
        smell_tensor = self.smell_data[ingredient][sample_idx].clone()
        return smell_tensor, label_idx


class LinearProbeModel(nn.Module):
    """
    Linear probe on top of frozen smell encoder.
    
    Args:
        smell_encoder: Frozen smell encoder from pretrained model
        feature_dim: Dimension of smell encoder output
        num_classes: Number of ingredient classes
    """
    
    def __init__(self, smell_encoder, feature_dim, num_classes):
        super().__init__()
        self.smell_encoder = smell_encoder
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Freeze smell encoder
        for param in self.smell_encoder.parameters():
            param.requires_grad = False
        
        # Set to eval mode to disable dropout, batchnorm, etc.
        self.smell_encoder.eval()
        
        console.print("[dim]→ Smell encoder frozen (eval mode)[/dim]")
        console.print(f"[dim]→ Linear classifier: {feature_dim} → {num_classes}[/dim]")
    
    def train(self, mode=True):
        """Override train to keep smell_encoder in eval mode"""
        super().train(mode)
        # Always keep smell_encoder in eval mode
        self.smell_encoder.eval()
        return self
    
    def forward(self, sensor_data, lengths=None):
        """
        Forward pass.
        
        Args:
            sensor_data: [B, T, F] sensor sequences
            lengths: [B] actual lengths (optional)
        
        Returns:
            logits: [B, num_classes]
        """
        with torch.no_grad():
            # Extract features from frozen backbone (before projection)
            features = self.smell_encoder(sensor_data, lengths)  # [B, D]
        
        # Classify
        logits = self.classifier(features)  # [B, num_classes]
        return logits


def load_pretrained_model(checkpoint_path, config_path, device):
    """
    Load pretrained Visual-Odor model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config_path: Path to YAML config used at training time
        device: Device to load model on

    Returns:
        smell_encoder: Frozen smell encoder
        config: Model configuration
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config YAML not found: {config_path}")

    console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")
    console.print(f"[cyan]Loading training config: {config_path}[/cyan]")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Load training hyperparameters from the YAML config used at training time.
    with open(config_path, 'r') as f:
        train_args = yaml.safe_load(f) or {}

    console.print(f"  [dim]Training configuration loaded from YAML[/dim]")
    
    # Map vision_encoder to vision_backbone and vision_model_name
    if 'vision_encoder' in train_args:
        vision_encoder = train_args['vision_encoder']
        if vision_encoder in ['clip_ViT-L-14', 'clip_ViT-B-16']:
            vision_backbone = 'clip'
            vision_model_name = vision_encoder.replace('clip_', '')
        else:  # dino variants
            vision_backbone = 'dino'
            vision_model_name = vision_encoder
    else:
        vision_backbone = train_args.get('vision_backbone', 'dino')
        vision_model_name = train_args.get('vision_model_name', 'dinov3_vits16')
    
    # Override inaccessible paths
    vision_dino_repo_local = train_args.get('vision_dino_repo_local', 'ext_repos/dinov3')
    if 'vision_dino_repo_local' in train_args:
        original_path = train_args['vision_dino_repo_local']
        if not os.path.exists(original_path):
            vision_dino_repo_local = 'ext_repos/dinov3'
            console.print(f"  [yellow]⚠ DINOv3 repo path overridden: {original_path} → {vision_dino_repo_local}[/yellow]")
    
    # Create model config from training args
    config = ModelConfig(
        # Vision settings
        vision_backbone=vision_backbone,
        vision_model_name=vision_model_name,
        vision_clip_pretrained=train_args.get('vision_clip_pretrained', 'datacomp_xl_s13b_b90k'),
        vision_dino_repo_local=vision_dino_repo_local,
        vision_forward_option=train_args['vision_forward_option'],
        vision_projection_type=train_args['vision_projection_type'],
        
        # Smell settings
        smell_input_dim=train_args.get('smell_input_dim', 6),  # Will be verified against data
        smell_model_dim=train_args['smell_model_dim'],
        smell_num_heads=train_args['smell_num_heads'],
        smell_num_layers=train_args['smell_num_layers'],
        smell_forward_option=train_args['smell_forward_option'],
        smell_use_positional_encoding=train_args.get('smell_use_positional_encoding', True),
        smell_projection_type=train_args['smell_projection_type'],
        
        # Projection
        target_embedding_dim=train_args['embed_dim'],
        use_norm=train_args.get('use_norm', True),
        dropout=train_args.get('dropout', 0.1),
    )
    
    # Create model
    model = VisualOdorModel(config).to(device)
    
    # Load state dict
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    
    console.print(f"[green]✓ Loaded pretrained model (epoch {checkpoint['epoch']})[/green]")
    
    # Print key config values
    console.print(f"  [dim]Model configuration:[/dim]")
    console.print(f"  [dim]  Vision - backbone: {config.vision_backbone}, model: {config.vision_model_name}[/dim]")
    console.print(f"  [dim]  Vision - forward: {config.vision_forward_option}, projection: {config.vision_projection_type}[/dim]")
    console.print(f"  [dim]  Smell - input_dim: {config.smell_input_dim}, model_dim: {config.smell_model_dim}[/dim]")
    console.print(f"  [dim]  Smell - layers: {config.smell_num_layers}, forward: {config.smell_forward_option}[/dim]")
    console.print(f"  [dim]  Smell - projection: {config.smell_projection_type}[/dim]")
    console.print(f"  [dim]  Target embedding dim: {config.target_embedding_dim}[/dim]")
    
    return model.smell_encoder, config, train_args


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for batch_idx, (sensor_data, labels) in enumerate(dataloader):
        sensor_data = sensor_data.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(sensor_data)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        
        # Top-1 accuracy
        _, predicted = logits.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    acc_top1 = 100. * correct_top1 / total
    acc_top5 = 100. * correct_top5 / total
    
    return avg_loss, acc_top1, acc_top5


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for sensor_data, labels in dataloader:
        sensor_data = sensor_data.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(sensor_data)
        loss = criterion(logits, labels)
        
        # Metrics
        total_loss += loss.item()
        
        # Top-1 accuracy
        _, predicted = logits.max(1)
        correct_top1 += predicted.eq(labels).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(5, dim=1, largest=True, sorted=True)
        correct_top5 += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
        
        total += labels.size(0)
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())
    
    avg_loss = total_loss / len(dataloader)
    acc_top1 = 100. * correct_top1 / total
    acc_top5 = 100. * correct_top5 / total
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    macro_f1 = 100. * f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return avg_loss, acc_top1, acc_top5, macro_f1


@torch.no_grad()
def save_per_window_predictions(model, smell_data, le, device, save_path):
    """
    Save per-window top-1 predictions for each ingredient.

    Args:
        model: Trained LinearProbeModel
        smell_data: dict[ingredient, list[Tensor(T,F)]]
        le: LabelEncoder
        device: torch device
        save_path: Path to save JSON file
    """
    model.eval()

    predictions = {}

    for ingredient in sorted(smell_data.keys()):
        preds = []
        tensors = smell_data[ingredient]

        for tensor in tensors:
            sensor_input = tensor.unsqueeze(0).to(device)  # [1, T, F]
            logits = model(sensor_input)  # [1, num_classes]
            pred_idx = logits.argmax(dim=1).item()
            pred_label = le.inverse_transform([pred_idx])[0]
            preds.append(pred_label)

        predictions[ingredient] = preds

    with open(save_path, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Print summary
    total_windows = sum(len(v) for v in predictions.values())
    correct = sum(1 for ing, preds in predictions.items() for p in preds if p == ing)
    acc = 100.0 * correct / total_windows if total_windows > 0 else 0.0
    console.print(f"[green]\u2713 Per-window predictions saved to: {save_path}[/green]")
    console.print(f"  [cyan]Total windows: {total_windows}, Correct: {correct}, Accuracy: {acc:.2f}%[/cyan]")

    return predictions


def main(args):
    """Main training function."""
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.print(Panel.fit(
        f"[bold cyan]Linear Probing for Smell Encoder[/bold cyan]\n"
        f"Device: {device}",
        border_style="cyan",
        box=box.DOUBLE
    ))
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. Load Pretrained Model
    # ========================================
    console.print("\n[bold green]│[/bold green] [1/6] Loading pretrained model...")
    smell_encoder, model_config, train_args = load_pretrained_model(args.checkpoint, args.config, device)
    
    # Apply preprocessing settings from training args
    console.print(f"  [dim]Applying preprocessing from training args:[/dim]")
    args.diff_periods = train_args.get('diff_periods', args.diff_periods)
    args.window_size = train_args.get('window_size', args.window_size)
    args.stride = train_args.get('stride', args.stride)
    args.removed_columns = train_args.get('removed_columns', args.removed_columns)
    args.max_seq_len = train_args.get('max_seq_len', args.max_seq_len)
    
    console.print(f"  [dim]  - diff_periods: {args.diff_periods}[/dim]")
    console.print(f"  [dim]  - window_size: {args.window_size}[/dim]")
    console.print(f"  [dim]  - stride: {args.stride}[/dim]")
    console.print(f"  [dim]  - removed_columns: {args.removed_columns}[/dim]")
    
    # ========================================
    # 2. Load Data
    # ========================================
    console.print("\n[bold green]│[/bold green] [2/6] Loading smell data...")
    
    # Load label encoder
    le = create_label_encoder_from_json(args.label_json)
    num_classes = len(le.classes_)
    console.print(f"  [cyan]Number of classes: {num_classes}[/cyan]")
    
    # Load training data
    train_data = load_sensor_data(
        args.smell_train_dir,
        removed_filtered_columns=args.removed_columns,
    )
    console.print(f"  [cyan]Loaded {len(train_data)} ingredients (train)[/cyan]")
    
    # Load test data
    test_data = load_sensor_data(
        args.smell_test_dir,
        removed_filtered_columns=args.removed_columns,
    )
    console.print(f"  [cyan]Loaded {len(test_data)} ingredients (test)[/cyan]")
    
    # Apply preprocessing
    if args.diff_periods is not None:
        console.print(f"  [cyan]Applying diff with periods={args.diff_periods}[/cyan]")
        train_data = diff_data_like(train_data, periods=args.diff_periods)
        test_data = diff_data_like(test_data, periods=args.diff_periods)
    
    # Create sliding windows
    if args.window_size is not None:
        console.print(f"  [cyan]Creating sliding windows (size={args.window_size}, stride={args.stride})[/cyan]")
        X_train, y_train = make_sliding_window_dataset(
            train_data, le, 
            window_size=args.window_size, 
            stride=args.stride
        )
        X_test, y_test = make_sliding_window_dataset(
            test_data, le, 
            window_size=args.window_size, 
            stride=args.stride
        )
        
        console.print(f"  [cyan]Train windows: {X_train.shape}, Test windows: {X_test.shape}[/cyan]")
        
        # Convert to dict format (same as main_pretrain)
        smell_train = group_windows_by_ingredient(X_train, y_train, le)
        smell_test = group_windows_by_ingredient(X_test, y_test, le)
        
        console.print(f"  [cyan]Grouped into {len(smell_train)} train ingredients, {len(smell_test)} test ingredients[/cyan]")
        
    else:
        # Use full sequences (pad/truncate to max_seq_len)
        console.print(f"  [cyan]Using full sequences (max_len={args.max_seq_len})[/cyan]")
        
        smell_train = {}
        for ingredient, dfs in train_data.items():
            smell_train[ingredient] = []
            for df in dfs:
                seq = df.values
                # Pad or truncate
                if len(seq) < args.max_seq_len:
                    pad = np.zeros((args.max_seq_len - len(seq), seq.shape[1]))
                    seq = np.vstack([seq, pad])
                else:
                    seq = seq[:args.max_seq_len]
                smell_train[ingredient].append(torch.tensor(seq, dtype=torch.float32))
        
        smell_test = {}
        for ingredient, dfs in test_data.items():
            smell_test[ingredient] = []
            for df in dfs:
                seq = df.values
                if len(seq) < args.max_seq_len:
                    pad = np.zeros((args.max_seq_len - len(seq), seq.shape[1]))
                    seq = np.vstack([seq, pad])
                else:
                    seq = seq[:args.max_seq_len]
                smell_test[ingredient].append(torch.tensor(seq, dtype=torch.float32))
        
        console.print(f"  [cyan]Processed {len(smell_train)} train ingredients, {len(smell_test)} test ingredients[/cyan]")
    
    # Get sample to infer smell_input_dim
    sample_ingredient = list(smell_train.keys())[0]
    sample_tensor = smell_train[sample_ingredient][0]
    inferred_smell_input_dim = sample_tensor.shape[-1]
    console.print(f"  [cyan]Inferred smell input dim: {inferred_smell_input_dim}[/cyan]")
    
    # Print data statistics
    total_train_samples = sum(len(tensors) for tensors in smell_train.values())
    total_test_samples = sum(len(tensors) for tensors in smell_test.values())
    console.print(f"  [cyan]Total samples - Train: {total_train_samples}, Test: {total_test_samples}[/cyan]")
    
    # ========================================
    # 3. Create Datasets and Dataloaders
    # ========================================
    console.print("\n[bold green]│[/bold green] [3/6] Creating dataloaders...")
    
    train_dataset = SmellClassificationDataset(smell_train, le)
    test_dataset = SmellClassificationDataset(smell_test, le)
    
    console.print(f"  [cyan]Train dataset: {len(train_dataset)} samples[/cyan]")
    console.print(f"  [cyan]Test dataset: {len(test_dataset)} samples[/cyan]")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    console.print(f"  [cyan]Train batches: {len(train_loader)}, Test batches: {len(test_loader)}[/cyan]")
    
    # ========================================
    # 4. Create Linear Probe Model
    # ========================================
    console.print("\n[bold green]│[/bold green] [4/6] Creating linear probe model...")
    
    # Verify smell_input_dim matches data
    if inferred_smell_input_dim != model_config.smell_input_dim:
        console.print(f"  [yellow]⚠ Warning: Data has {inferred_smell_input_dim} features, "
                     f"but checkpoint model expects {model_config.smell_input_dim}[/yellow]")
        console.print(f"  [yellow]  This might cause dimension mismatch errors![/yellow]")
    
    # Get feature dimension from smell encoder backbone
    if model_config.smell_projection_type in ('aligner', 'residual_mlp'):
        feature_dim = model_config.target_embedding_dim
    else:   
        feature_dim = model_config.smell_model_dim
    
    console.print(f"  [cyan]Smell backbone output dim: {feature_dim}[/cyan]")
    
    model = LinearProbeModel(
        smell_encoder=smell_encoder,
        feature_dim=feature_dim,
        num_classes=num_classes,
    ).to(device)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"  [cyan]Total params: {total_params:,}[/cyan]")
    console.print(f"  [cyan]Trainable params: {trainable_params:,}[/cyan]")
    
    # ========================================
    # 5. Setup Training
    # ========================================
    console.print("\n[bold green]│[/bold green] [5/6] Setting up training...")
    
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'adam':
        optimizer = Adam(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = SGD(model.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    console.print(f"  [cyan]Optimizer: {args.optimizer}[/cyan]")
    console.print(f"  [cyan]Learning rate: {args.lr}[/cyan]")
    console.print(f"  [cyan]Weight decay: {args.weight_decay}[/cyan]")
    console.print(f"  [cyan]Epochs: {args.epochs}[/cyan]")
    
    # ========================================
    # 6. Training Loop
    # ========================================
    console.print("\n[bold green]│[/bold green] [6/6] Training...")
    console.print()
    
    best_test_acc = 0.0
    best_f1_at_best_acc = 0.0
    best_epoch = -1
    results = []
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc1, train_acc5 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate
        test_loss, test_acc1, test_acc5, test_f1 = evaluate(
            model, test_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Print progress
        console.print(
            f"Epoch [{epoch:3d}/{args.epochs}] "
            f"Train: loss={train_loss:.4f} top1={train_acc1:.2f}% top5={train_acc5:.2f}% | "
            f"Test: loss={test_loss:.4f} top1={test_acc1:.2f}% top5={test_acc5:.2f}% f1={test_f1:.2f}% | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # Save results
        results.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc_top1': train_acc1,
            'train_acc_top5': train_acc5,
            'test_loss': test_loss,
            'test_acc_top1': test_acc1,
            'test_acc_top5': test_acc5,
            'test_f1_macro': test_f1,
            'time': epoch_time,
        })
        
        # Save best model
        # Update if: acc improved, OR acc tied and F1 improved
        if test_acc1 > best_test_acc or (test_acc1 == best_test_acc and test_f1 > best_f1_at_best_acc):
            best_test_acc = test_acc1
            best_f1_at_best_acc = test_f1
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc_top1': test_acc1,
                'test_acc_top5': test_acc5,
                'test_f1_macro': test_f1,
                'test_loss': test_loss,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            console.print(f"  [green]✓ Saved best model (test top-1: {test_acc1:.2f}%, top-5: {test_acc5:.2f}%, f1: {test_f1:.2f}%)[/green]")
    
    # ========================================
    # Final Results
    # ========================================
    console.print()
    console.print(Panel.fit(
        f"[bold green]Training Complete![/bold green]\n"
        f"Best Test Accuracy (top-1): {best_test_acc:.2f}%\n"
        f"Best Test F1 (macro):       {best_f1_at_best_acc:.2f}%\n"
        f"(at epoch {best_epoch})",
        border_style="green",
        box=box.DOUBLE
    ))
    
    # Save results with summary
    summary = {
        'best_acc_top1': best_test_acc,
        'best_f1_macro': best_f1_at_best_acc,
        'best_epoch': best_epoch,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump({'summary': summary, 'epochs': results}, f, indent=2)
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc_top1': test_acc1,
        'test_acc_top5': test_acc5,
        'test_f1_macro': test_f1,
        'test_loss': test_loss,
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    console.print(f"[cyan]Results saved to: {args.output_dir}[/cyan]")


def get_args_parser():
    parser = argparse.ArgumentParser('Linear Probing for Smell Encoder', add_help=False)
    
    # Training YAML config (required - the same file the checkpoint was trained with)
    parser.add_argument('--config', required=True, type=str,
                        help='Path to YAML config from training (e.g., configs/SeeandSniff.yaml)')
    
    # Checkpoint
    parser.add_argument('--checkpoint', required=True, type=str,
                        help='Path to pretrained model checkpoint')
    
    # Data
    parser.add_argument('--smell_train_dir', default='datasets/SmellNet/base_data/training', type=str,
                        help='Path to training smell data directory')
    parser.add_argument('--smell_test_dir', default='datasets/SmellNet/base_data/testing', type=str,
                        help='Path to testing smell data directory')
    parser.add_argument('--label_json', default='metadata/ingredient_labels.json', type=str,
                        help='Path to ingredient labels JSON')
    
    # Data preprocessing (should match pretraining settings)
    # Note: These will be overridden by --config YAML values
    parser.add_argument('--diff_periods', default=50, type=int,
                        help='Diff periods for smell data (overridden by --config)')
    parser.add_argument('--window_size', default=40, type=int,
                        help='Sliding window size (overridden by --config)')
    parser.add_argument('--stride', default=20, type=int,
                        help='Sliding window stride (overridden by --config)')
    parser.add_argument('--max_seq_len', default=1000, type=int,
                        help='Max sequence length when not using windows (overridden by --config)')
    parser.add_argument('--removed_columns',
                        default=['Benzene', 'Temperature', 'Pressure', 'Humidity', 'Gas_Resistance', 'Altitude'],
                        type=list,
                        help='Columns to remove from smell data (overridden by --config)')
    
    # Training
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--epochs', default=50, type=int,
                        help='Number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float,
                        help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='Weight decay')
    parser.add_argument('--optimizer', default='adam', type=str, choices=['adam', 'sgd'],
                        help='Optimizer type')
    
    # System
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of dataloader workers')
    parser.add_argument('--output_dir', default='outputs/linear_probing', type=str,
                        help='Output directory for results')
    
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Linear Probing', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
