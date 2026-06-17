"""
Main training script for Visual-Odor cross-modal pretraining.

Features:
- Unified argument names with model/loss/dataset/engine modules
- Single-GPU training
- Flexible data preprocessing (diff, sliding window)
- WandB logging
- Comprehensive configuration via YAML
"""

import argparse
import datetime
import io
import json
import re
import sys
import yaml
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.amp import GradScaler

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    console.print("[yellow]⚠ wandb not installed. Install with: pip install wandb[/yellow]")

from model import VisualOdorModel, ModelConfig
from loss import VisualOdorLoss
from dataset import VisualOdorDataset, group_windows_by_ingredient
from load_data import (
    load_vision_data,
    load_sensor_data,
    create_label_encoder_from_json,
    RGB_PREPROCESS,
    RGB_AUGMENTS,
    diff_data_like,
    make_sliding_window_dataset,
)
from engine_pretrain import train_one_epoch, evaluate
from misc import (
    setup_seed,
    save_checkpoint,
    load_checkpoint,
    load_smellnet_checkpoint,
    compute_param_stats,
    print_model_tables,
)


def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Odor Pretraining', add_help=False)

    # Config file
    parser.add_argument('--config', default=None, type=str,
                        help='Path to YAML config file (overrides other arguments)')

    # ========== Data Parameters ==========
    parser.add_argument('--vision_train_json', default='metadata/train_metadata.json', type=str,
                        help='Path to training vision metadata JSON')
    parser.add_argument('--vision_test_json', default='metadata/test_metadata.json', type=str,
                        help='Path to testing vision metadata JSON')
    parser.add_argument('--vision_train_dir', default='datasets/train', type=str,
                        help='Base directory for training vision data')
    parser.add_argument('--vision_test_dir', default='datasets/test', type=str,
                        help='Base directory for testing vision data')
    parser.add_argument('--smell_train_dir', default='datasets/SmellNet/base_data/training', type=str,
                        help='Path to training smell data directory')
    parser.add_argument('--smell_test_dir', default='datasets/SmellNet/base_data/testing', type=str,
                        help='Path to testing smell data directory')
    parser.add_argument('--label_json', default='metadata/ingredient_labels.json', type=str,
                        help='Path to ingredient labels JSON')

    # Smell data preprocessing
    parser.add_argument('--diff_periods', default=None, type=int,
                        help='Diff periods for smell data (None for no diff)')
    parser.add_argument('--window_size', default=None, type=int,
                        help='Sliding window size for smell data (None for no windowing)')
    parser.add_argument('--stride', default=None, type=int,
                        help='Sliding window stride (required if window_size is set)')
    parser.add_argument('--removed_columns', default=['Benzene', 'Temperature', 'Pressure', 'Humidity', 'Gas_Resistance', 'Altitude'],
                        type=list, help='Columns to remove from smell data')
    parser.add_argument('--max_seq_len', default=1000, type=int,
                        help='Maximum sequence length for smell data (used when window_size is None)')

    # Dataset
    parser.add_argument('--pairing_mode', default='cycled_shuffle', type=str,
                        choices=['naive_random', 'fixed_random', 'cycled_shuffle'],
                        help='Vision-smell pairing strategy')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--pin_mem', default=True, type=bool,
                        help='Pin CPU memory in DataLoader')

    # ========== Model Parameters ==========
    # Vision encoder
    parser.add_argument('--vision_encoder', default='clip_ViT-L-14', type=str,
                        choices=['clip_ViT-L-14', 'clip_ViT-B-16', 'dinov3_vits16', 'dinov3_vitb16'],
                        help='Vision encoder architecture')
    parser.add_argument('--vision_forward_option', default='cls_token', type=str,
                        choices=['cls_token', 'spatial_tokens'],
                        help='Vision encoder forward option')
    parser.add_argument('--vision_projection_type', default='aligner', type=str,
                        choices=['aligner', 'none'],
                        help='Vision projection type')
    parser.add_argument('--vision_freeze_backbone', default=False, type=bool,
                        help='Freeze vision backbone')
    parser.add_argument('--vision_freeze_projection', default=False, type=bool,
                        help='Freeze vision projection layer')

    # Smell encoder
    parser.add_argument('--smell_forward_option', default='cls_token', type=str,
                        choices=['cls_token', 'spatial_tokens'],
                        help='Smell encoder forward option')
    parser.add_argument('--smell_projection_type', default='aligner', type=str,
                        choices=['aligner', 'none'],
                        help='Smell projection type')
    parser.add_argument('--smell_model_dim', default=256, type=int,
                        help='Model dimension for smell transformer')
    parser.add_argument('--smell_num_heads', default=8, type=int,
                        help='Number of attention heads in smell transformer')
    parser.add_argument('--smell_num_layers', default=4, type=int,
                        help='Number of transformer layers in smell encoder')
    parser.add_argument('--smell_freeze_backbone', default=False, type=bool,
                        help='Freeze smell backbone')
    parser.add_argument('--smell_freeze_projection', default=False, type=bool,
                        help='Freeze smell projection layer')
    parser.add_argument('--smell_pretrained_checkpoint', default=None, type=str,
                        help='Path to pretrained SmellNet checkpoint (.pt file)')

    # Common dimension
    parser.add_argument('--embed_dim', default=512, type=int,
                        help='Final embedding dimension for projection')

    # ========== Loss Parameters ==========
    parser.add_argument('--loss_forward_mode', default='global', type=str,
                        choices=['global', 'local'],
                        help='Loss forward mode (must match vision_forward_option)')
    parser.add_argument('--spatial_pool', default='mean', type=str,
                        choices=['mean', 'max'],
                        help='Spatial pooling method for spatial mode')
    parser.add_argument('--retrieval_mode', default='ingredientwise', type=str,
                        choices=['samplewise', 'ingredientwise'],
                        help='Retrieval evaluation mode')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='Initial temperature for logit scaling')

    # ========== Optimizer Parameters ==========
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Base learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    parser.add_argument('--lr_scale_vision', default=1.0, type=float,
                        help='Learning rate scale for vision encoder')
    parser.add_argument('--lr_scale_smell', default=1.0, type=float,
                        help='Learning rate scale for smell encoder')
    parser.add_argument('--lr_scale_aligner', default=1.0, type=float,
                        help='Learning rate scale for aligner')
    parser.add_argument('--clip_grad', default=None, type=float,
                        help='Gradient clipping max norm (None for no clipping)')

    # ========== Training Parameters ==========
    parser.add_argument('--epochs', default=100, type=int,
                        help='Number of training epochs')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Number of warmup epochs')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Start epoch (for resume)')

    # ========== System Parameters ==========
    parser.add_argument('--output_dir', default='outputs/pretrain', type=str,
                        help='Path to save checkpoints and logs')
    parser.add_argument('--log_dir', default='outputs/logs', type=str,
                        help='Path for tensorboard logs')
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training')
    parser.add_argument('--seed', default=42, type=int,
                        help='Random seed')
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Checkpoint save frequency (epochs)')

    # ========== WandB ==========
    parser.add_argument('--use_wandb', default=False, type=bool,
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', default='SeeandSniff', type=str,
                        help='WandB project name')
    parser.add_argument('--experiment_name', default='experiment', type=str,
                        help='Experiment name for logging')
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU ID to use (e.g., "0")')

    return parser


def main(args):
    """
    Main training function (single-GPU).

    Usage:
        python main_pretrain.py --config configs/SeeandSniff.yaml --gpu "0"
    """
    # ========================================
    # GPU Selection
    # ========================================
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    console.print(f"[cyan]ℹ CUDA_VISIBLE_DEVICES set to: {args.gpu}[/cyan]")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    console.print(Panel.fit(
        f"[bold cyan]Visual-Odor Cross-Modal Pretraining[/bold cyan]\nSingle GPU training",
        border_style="cyan",
        box=box.DOUBLE
    ))

    # ========================================
    # Setup
    # ========================================
    console.print("\n[bold green]│[/bold green] [1/8] Setting up environment...")

    # Random seed
    setup_seed(args.seed)

    # Output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to log files (strip ANSI codes for file output)
    _log_file = open(os.path.join(args.log_dir, 'train.log'), 'w')
    _ansi_re = re.compile(r'\x1b\[[0-9;]*m')
    class _Tee(io.TextIOBase):
        def __init__(self, terminal, log):
            self.terminal = terminal
            self.log = log
        def write(self, data):
            self.terminal.write(data)
            self.terminal.flush()
            self.log.write(_ansi_re.sub('', data))
            self.log.flush()
            return len(data)
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    sys.stdout = _Tee(sys.__stdout__, _log_file)
    sys.stderr = _Tee(sys.__stderr__, _log_file)

    env_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    env_table.add_column(style="dim")
    env_table.add_column()
    env_table.add_row("Device", f"[cyan]{device}[/cyan]")
    env_table.add_row("Random seed", f"[cyan]{args.seed}[/cyan]")
    env_table.add_row("Output directory", f"[cyan]{args.output_dir}[/cyan]")
    env_table.add_row("Log directory", f"[cyan]{args.log_dir}[/cyan]")
    console.print(env_table)

    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # WandB
    if args.use_wandb:
        if WANDB_AVAILABLE:
            wandb.init(
                project=args.wandb_project,
                name=args.experiment_name,
                config=vars(args),
            )
            console.print(f"  [green]✓[/green] WandB initialized: [cyan]{args.wandb_project}/{args.experiment_name}[/cyan]")
        else:
            console.print(f"  [yellow]⚠ WandB requested but not available[/yellow]")

    # ========================================
    # Data Loading
    # ========================================
    console.print("\n[bold green]│[/bold green] [2/8] Loading data...")

    # Label encoder
    le = create_label_encoder_from_json(args.label_json)
    console.print(f"  [green]✓[/green] Loaded [cyan]{len(le.classes_)}[/cyan] ingredient labels")

    # Vision data
    console.print(f"  [dim]Loading vision data...[/dim]")
    vision_train = load_vision_data(
        json_path=args.vision_train_json,
        base_image_dir=args.vision_train_dir,
        transform_rgb=RGB_AUGMENTS,
    )
    vision_test = load_vision_data(
        json_path=args.vision_test_json,
        base_image_dir=args.vision_test_dir,
        transform_rgb=RGB_PREPROCESS,
    )

    total_train_images = sum(len(imgs) for imgs in vision_train.values())
    console.print(f"  [green]✓[/green] [TRAIN] Loaded [cyan]{len(vision_train)}[/cyan] ingredients, [cyan]{total_train_images}[/cyan] images")

    total_test_images = sum(len(imgs) for imgs in vision_test.values())
    console.print(f"  [green]✓[/green] [TEST] Loaded [cyan]{len(vision_test)}[/cyan] ingredients, [cyan]{total_test_images}[/cyan] images")

    # Smell data
    console.print(f"  [dim]Loading smell data...[/dim]")
    sensor_train = load_sensor_data(
        data_path=args.smell_train_dir,
        removed_filtered_columns=args.removed_columns,
    )
    sensor_test = load_sensor_data(
        data_path=args.smell_test_dir,
        removed_filtered_columns=args.removed_columns,
    )

    # Apply diff if specified
    if args.diff_periods is not None:
        console.print(f"  [dim]Applying diff with periods={args.diff_periods}...[/dim]")
        sensor_train = diff_data_like(sensor_train, periods=args.diff_periods)
        sensor_test = diff_data_like(sensor_test, periods=args.diff_periods)

    # Create windowed dataset
    if args.window_size is None:
        # Use the longest sequence as window size (no actual windowing)
        # Consider both train and test to ensure consistent window size
        max_train_len = max(max(len(df) for df in dfs) for dfs in sensor_train.values())
        max_test_len = max(max(len(df) for df in dfs) for dfs in sensor_test.values())
        max_len = max(max_train_len, max_test_len)
        window_size = min(max_len, args.max_seq_len)
        stride = window_size  # No overlap
        console.print(f"  [dim]Using full sequences (train_max={max_train_len}, test_max={max_test_len}, window={window_size}, stride={stride})...[/dim]")
    else:
        if args.stride is None:
            raise ValueError("stride must be specified when window_size is set")
        window_size = args.window_size
        stride = args.stride
        console.print(f"  [dim]Creating sliding windows (window={window_size}, stride={stride})...[/dim]")

    # Train windows
    X_train_windows, y_train_windows = make_sliding_window_dataset(
        sensor_train,
        le,
        window_size=window_size,
        stride=stride
    )

    console.print(f"  [green]✓[/green] [TRAIN] Created [cyan]{len(X_train_windows)}[/cyan] windows (avg [cyan]{len(X_train_windows)/50:.2f}[/cyan] per ingredient)")
    console.print(f"    Window shape: [cyan]{X_train_windows.shape}[/cyan]  [dim]# (N, window_size, channels)[/dim]")

    # Convert windowed data back to dict format
    smell_train = group_windows_by_ingredient(X_train_windows, y_train_windows, le)
    total_train_samples = sum(len(tensors) for tensors in smell_train.values())
    console.print(f"  [green]✓[/green] [TRAIN] Grouped into [cyan]{len(smell_train)}[/cyan] ingredients, [cyan]{total_train_samples}[/cyan] windows")

    # Test windows
    X_test_windows, y_test_windows = make_sliding_window_dataset(
        sensor_test,
        le,
        window_size=window_size,
        stride=stride
    )

    console.print(f"  [green]✓[/green] [TEST] Created [cyan]{len(X_test_windows)}[/cyan] windows (avg [cyan]{len(X_test_windows)/50:.2f}[/cyan] per ingredient)")
    console.print(f"    Window shape: [cyan]{X_test_windows.shape}[/cyan]  [dim]# (N, window_size, channels)[/dim]")

    # Convert windowed data back to dict format
    smell_test = group_windows_by_ingredient(X_test_windows, y_test_windows, le)
    total_test_samples = sum(len(tensors) for tensors in smell_test.values())
    console.print(f"  [green]✓[/green] [TEST] Grouped into [cyan]{len(smell_test)}[/cyan] ingredients, [cyan]{total_test_samples}[/cyan] windows")

    # Dataset
    console.print(f"  [dim]Creating train dataset (pairing_mode={args.pairing_mode})...[/dim]")
    train_dataset = VisualOdorDataset(
        vision_data=vision_train,
        smell_data=smell_train,
        le=le,
        pairing_mode=args.pairing_mode,
        seed=args.seed,
    )

    console.print(f"  [dim]Creating test dataset (pairing_mode=cycled_shuffle)...[/dim]")
    test_dataset = VisualOdorDataset(
        vision_data=vision_test,
        smell_data=smell_test,
        le=le,
        pairing_mode='cycled_shuffle',  # Fixed pairing for evaluation
        seed=args.seed,
    )

    # DataLoader — seed each worker process and use a per-run torch.Generator
    # so the shuffle order and any numpy/random use inside __getitem__ are
    # deterministic across runs with the same --seed.
    import random as _py_random

    def _worker_init_fn(worker_id):
        base = torch.initial_seed() % 2**32
        np.random.seed(base + worker_id)
        _py_random.seed(base + worker_id)

    train_gen = torch.Generator().manual_seed(args.seed)
    test_gen  = torch.Generator().manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        worker_init_fn=_worker_init_fn,
        generator=train_gen,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,  # Keep all samples for evaluation
        worker_init_fn=_worker_init_fn,
        generator=test_gen,
    )

    console.print(f"  [green]✓[/green] Train dataset created: [cyan]{len(train_dataset)}[/cyan] pairs")
    console.print(f"  [green]✓[/green] Train DataLoader created: [cyan]{len(train_loader)}[/cyan] batches")
    console.print(f"  [green]✓[/green] Test dataset created: [cyan]{len(test_dataset)}[/cyan] pairs")
    console.print(f"  [green]✓[/green] Test DataLoader created: [cyan]{len(test_loader)}[/cyan] batches")

    # ========================================
    # Model Creation
    # ========================================
    console.print("\n[bold green]│[/bold green] [3/8] Creating model...")

    # Determine smell input dim from data
    sample_ing = list(smell_train.keys())[0]
    sample_tensor = smell_train[sample_ing][0]
    inferred_smell_input_dim = sample_tensor.shape[-1]

    # Map vision encoder names
    if args.vision_encoder in ['clip_ViT-L-14', 'clip_ViT-B-16']:
        vision_backbone = 'clip'
        vision_model_name = args.vision_encoder.replace('clip_', '')
    else:  # dino_vits16 or dino_vitb16
        vision_backbone = 'dino'
        vision_model_name = args.vision_encoder
        # vision_model_name = args.vision_encoder.replace('dino_', 'dinov3_')

    config = ModelConfig(
        # Vision
        vision_backbone=vision_backbone,
        vision_model_name=vision_model_name,
        vision_forward_option=args.vision_forward_option,
        vision_projection_type=args.vision_projection_type,
        vision_freeze_backbone=args.vision_freeze_backbone,
        vision_freeze_projection=args.vision_freeze_projection,
        # Smell
        smell_input_dim=inferred_smell_input_dim,
        smell_model_dim=args.smell_model_dim,
        smell_num_heads=args.smell_num_heads,
        smell_num_layers=args.smell_num_layers,
        smell_forward_option=args.smell_forward_option,
        smell_projection_type=args.smell_projection_type,
        smell_freeze_backbone=args.smell_freeze_backbone,
        smell_freeze_projection=args.smell_freeze_projection,
        # Projection
        target_embedding_dim=args.embed_dim,
        # Temperature
        init_logit_scale=np.log(1 / args.temperature),
    )

    model = VisualOdorModel(config).to(device)

    # Load pretrained SmellNet checkpoint if specified
    if args.smell_pretrained_checkpoint:
        console.print(f"  [dim]Loading pretrained SmellNet checkpoint...[/dim]")

        checkpoint_info = load_smellnet_checkpoint(
            smell_encoder=model.smell_encoder,
            checkpoint_path=args.smell_pretrained_checkpoint,
            device=device,
            strict=False  # Allow missing keys (projection layer is new)
        )

        console.print(f"  [green]✓[/green] Loaded pretrained SmellNet weights")

        # Validate config compatibility
        ckpt_config = checkpoint_info['model_config']
        ckpt_dim = ckpt_config.get('tf_dim')
        ckpt_layers = ckpt_config.get('tf_layers')
        ckpt_heads = ckpt_config.get('tf_heads')
        ckpt_input_dim = ckpt_config.get('num_features')

        warnings = []
        if ckpt_dim != args.smell_model_dim:
            warnings.append(f"model_dim mismatch: checkpoint={ckpt_dim}, args={args.smell_model_dim}")
        if ckpt_layers != args.smell_num_layers:
            warnings.append(f"num_layers mismatch: checkpoint={ckpt_layers}, args={args.smell_num_layers}")
        if ckpt_heads != args.smell_num_heads:
            warnings.append(f"num_heads mismatch: checkpoint={ckpt_heads}, args={args.smell_num_heads}")
        if ckpt_input_dim != inferred_smell_input_dim:
            warnings.append(f"input_dim mismatch: checkpoint={ckpt_input_dim}, data={inferred_smell_input_dim}")

        if warnings:
            console.print(f"  [yellow]⚠ Configuration warnings:[/yellow]")
            for w in warnings:
                console.print(f"    [yellow]- {w}[/yellow]")

    console.print(f"  [green]✓[/green] Model created")
    stats = compute_param_stats(model)
    print_model_tables(console, args, inferred_smell_input_dim, stats)

    # ========================================
    # Loss Function
    # ========================================
    console.print("\n[bold green]│[/bold green] [4/8] Creating loss function...")

    # Validate loss_forward_mode matches vision_forward_option
    expected_forward_mode = 'local' if args.vision_forward_option == 'spatial_tokens' else 'global'
    if args.loss_forward_mode != expected_forward_mode:
        raise ValueError(
            f"loss_forward_mode='{args.loss_forward_mode}' must match "
            f"vision_forward_option='{args.vision_forward_option}' "
            f"(expected '{expected_forward_mode}')"
        )

    loss_fn = VisualOdorLoss(
        forward_mode=args.loss_forward_mode,
        spatial_pool=args.spatial_pool,
        retrieval_mode=args.retrieval_mode,
        label_json_path=args.label_json
    )

    # ========================================
    # [5/8] Optimizer
    # ========================================
    console.print("\n[bold green]│[/bold green] [5/8] Creating optimizer with WD separation...")

    def get_param_groups(model, base_lr, weight_decay, scales):
        """
        Build optimizer param groups with proper weight decay separation.

        Weight decay exclusion logic:
        - ndim < 2: biases, norms (BN/LN beta/gamma), scalars
        - Keywords: "bias", "norm", "bn", "ln", "cls_token", "pos_embed", etc.
        - Special: logit_scale, temperature

        Args:
            model: model instance
            base_lr: base learning rate
            weight_decay: weight decay coefficient
            scales: dict of lr_scale per module type
        """
        param_groups = []
        param_set = set()  # Track included params to avoid duplicates

        # Define no_decay keywords (learnable embeddings, tokens, norms, etc.)
        no_decay_keywords = (
            "bias", "norm", "bn", "ln", "gn",  # Bias & Norms
            "cls_token", "pos_embed", "positional", "mask_token",  # Learnable tokens/embeddings
            "relative_position", "gamma", "beta"  # Position biases & norm params
        )

        module_configs = [
            (model.vision_encoder.backbone, "vision_backbone", scales['vision']),
            (model.vision_encoder.projection, "vision_projection", scales['aligner']),
            (model.smell_encoder.backbone, "smell_backbone", scales['smell']),
            (model.smell_encoder.projection, "smell_projection", scales['aligner']),
        ]

        for module, name, lr_scale in module_configs:
            if module is None:
                continue

            decay = []
            no_decay = []

            for p_name, p in module.named_parameters():
                if not p.requires_grad:
                    continue

                # Check for duplicates
                if id(p) in param_set:
                    console.print(f"[yellow]⚠ Duplicate param detected: {name}.{p_name}[/yellow]")
                    continue
                param_set.add(id(p))

                # WD exclusion logic:
                # 1. ndim < 2 (bias, 1D norm params, scalars)
                # 2. Parameter name contains no_decay keywords
                if p.ndim < 2 or any(kw in p_name.lower() for kw in no_decay_keywords):
                    no_decay.append(p)
                else:
                    decay.append(p)

            if decay:
                param_groups.append({
                    "params": decay,
                    "lr": base_lr,
                    "lr_scale": lr_scale,
                    "weight_decay": weight_decay,
                    "name": f"{name}_decay"
                })
            if no_decay:
                param_groups.append({
                    "params": no_decay,
                    "lr": base_lr,
                    "lr_scale": lr_scale,
                    "weight_decay": 0.0,
                    "name": f"{name}_no_decay"
                })

        # Logit scale
        if hasattr(model, 'logit_scale') and model.logit_scale.requires_grad:
            if id(model.logit_scale) not in param_set:
                param_groups.append({
                    "params": [model.logit_scale],
                    "lr": base_lr,
                    "lr_scale": 1.0,
                    "weight_decay": 0.0,
                    "name": "logit_scale_no_decay"
                })
                param_set.add(id(model.logit_scale))

        # Validation: check for missing params
        all_trainable = {id(p) for p in model.parameters() if p.requires_grad}
        missing = all_trainable - param_set
        if missing:
            console.print(f"[red]⚠ WARNING: {len(missing)} trainable params NOT in optimizer![/red]")
            # Find names of missing params
            for n, p in model.named_parameters():
                if p.requires_grad and id(p) in missing:
                    console.print(f"  [red]Missing: {n} (shape={p.shape})[/red]")

        return param_groups

    lr_scales = {
        'vision': args.lr_scale_vision,
        'smell': args.lr_scale_smell,
        'aligner': args.lr_scale_aligner
    }
    param_groups = get_param_groups(model, args.lr, args.weight_decay, lr_scales)

    optimizer = torch.optim.AdamW(param_groups)

    console.print(f"  [green]✓[/green] Optimizer created (AdamW)")

    opt_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    opt_table.add_column(style="dim")
    opt_table.add_column()
    opt_table.add_row("Base LR", f"[cyan]{args.lr}[/cyan]")
    opt_table.add_row("Weight decay", f"[cyan]{args.weight_decay}[/cyan]")
    opt_table.add_row("LR scales", f"vision=[cyan]{args.lr_scale_vision}[/cyan], smell=[cyan]{args.lr_scale_smell}[/cyan], aligner=[cyan]{args.lr_scale_aligner}[/cyan]")

    # Param groups breakdown
    pg_table = Table(show_header=True, box=box.SIMPLE, padding=(0, 1))
    pg_table.add_column("Group", style="cyan")
    pg_table.add_column("# Params", justify="right")
    pg_table.add_column("LR Scale", justify="right")
    pg_table.add_column("WD", justify="right")
    pg_table.add_column("Example Params", style="dim")

    total_params_in_opt = 0
    for i, pg in enumerate(optimizer.param_groups):
        params = pg['params']
        num_params = sum(p.numel() for p in params)
        total_params_in_opt += num_params

        # Get sample param names from this group
        sample_names = []
        for n, p in model.named_parameters():
            if any(id(p) == id(pg_p) for pg_p in params):
                sample_names.append(n.split('.')[-1])  # Last component only
                if len(sample_names) >= 2:
                    break
        sample_str = ", ".join(sample_names) if sample_names else "-"

        pg_table.add_row(
            pg.get('name', f'group_{i}'),
            f"{num_params:,}",
            f"{pg.get('lr_scale', 1.0):.2f}",
            f"{pg.get('weight_decay', 0.0):.4f}",
            sample_str
        )

    # Validation row
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    match_status = "✓" if total_params_in_opt == total_trainable else f"✗ ({total_params_in_opt}/{total_trainable})"
    pg_table.add_section()
    pg_table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{total_params_in_opt:,}[/bold]",
        "",
        "",
        f"[bold]Validation: {match_status}[/bold]"
    )

    console.print(opt_table)
    console.print(pg_table)

    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == "cuda"))
    console.print(f"  [green]✓[/green] GradScaler created for mixed precision")

    # ========================================
    # Resume/Load Checkpoint
    # ========================================
    start_epoch = args.start_epoch
    best_loss = float('inf')

    if args.resume:
        console.print("\n[bold green]│[/bold green] [6/8] Loading checkpoint...")
        start_epoch, best_loss = load_checkpoint(
            model, optimizer, scaler, args.resume, device
        )
    else:
        console.print("\n[bold green]│[/bold green] [6/8] Starting from scratch (no checkpoint)")

    # ========================================
    # Training Loop
    # ========================================
    console.print("\n[bold green]│[/bold green] [7/8] Starting training...")

    train_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    train_table.add_column(style="dim")
    train_table.add_column()
    train_table.add_row("Epochs", f"[cyan]{start_epoch}[/cyan] → [cyan]{args.epochs}[/cyan]")
    train_table.add_row("Warmup epochs", f"[cyan]{args.warmup_epochs}[/cyan]")
    train_table.add_row("Evaluation frequency", f"every [cyan]{args.eval_freq}[/cyan] epochs")
    train_table.add_row("Save frequency", f"every [cyan]{args.save_freq}[/cyan] epochs")
    console.print(train_table)
    console.print()

    start_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        # Update dataset pairing
        train_dataset.on_epoch_start(epoch)

        # Train one epoch
        train_stats = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            args=args,
            log_writer=None,  # WandB handles logging
        )

        # WandB logging
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_stats['loss'],
                'train/lr': train_stats['lr'],
                'train/logit_scale': train_stats.get('logit_scale', 0),
            })

        # Evaluate
        val_stats = None
        if epoch % args.eval_freq == 0 or epoch == args.epochs - 1:
            console.print(f"  [dim]Running evaluation on test set...[/dim]")

            val_stats = evaluate(
                model=model,
                loss_fn=loss_fn,
                vision_test=vision_test,
                smell_test=smell_test,
                le=le,
                device=device,
                epoch=epoch,
                log_writer=None,
            )

            console.print(f"  [green]✓[/green] Test loss: [cyan]{val_stats['loss']:.4f}[/cyan]")
            if 'retrieval_acc' in val_stats:
                console.print(f"  [green]✓[/green] Retrieval R@1: [cyan]{val_stats['retrieval_acc']:.4f}[/cyan]")

            # WandB logging
            if args.use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    'epoch': epoch,
                    'val/loss': val_stats['loss'],
                }
                for k, v in val_stats.items():
                    if k not in ['loss', 'affinity_matrix']:
                        log_dict[f'val/{k}'] = v
                wandb.log(log_dict)

        # Save checkpoint
        # Check if best
        is_best = False
        if val_stats is not None and val_stats['loss'] < best_loss:
            best_loss = val_stats['loss']
            is_best = True

        # Save regular checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs - 1 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'args': args,
                'best_loss': best_loss,
                'train_stats': train_stats,
                'val_stats': val_stats
            }

            if is_best:
                save_checkpoint(checkpoint, is_best=True, output_dir=args.output_dir)
                console.print(f"  [green]✓[/green] Saved best checkpoint (loss: {best_loss:.4f})")

            if epoch % args.save_freq == 0 or epoch == args.epochs - 1:
                filename = f'checkpoint_epoch_{epoch}.pth'
                save_checkpoint(checkpoint, is_best=False, output_dir=args.output_dir, filename=filename)
                console.print(f"  [green]✓[/green] Saved checkpoint: {filename}")

        print()  # Empty line between epochs

    # ========================================
    # Final Evaluation
    # ========================================
    console.print("\n[bold green]│[/bold green] [8/8] Final evaluation on test set...")

    final_stats = evaluate(
        model=model,
        loss_fn=loss_fn,
        vision_test=vision_test,
        smell_test=smell_test,
        le=le,
        device=device,
        epoch=args.epochs,
        log_writer=None,
    )

    total_time = time.time() - start_time

    summary_table = Table(show_header=False, box=None, padding=(0, 2))
    summary_table.add_column(style="dim")
    summary_table.add_column()
    summary_table.add_row("Total time", f"[cyan]{datetime.timedelta(seconds=int(total_time))}[/cyan]")
    summary_table.add_row("Best validation loss", f"[cyan]{best_loss:.4f}[/cyan]")
    summary_table.add_row("Final validation loss", f"[cyan]{final_stats['loss']:.4f}[/cyan]")
    summary_table.add_row("Checkpoints saved to", f"[cyan]{args.output_dir}[/cyan]")
    summary_table.add_row("Logs saved to", f"[cyan]{args.log_dir}[/cyan]")

    console.print(Panel(
        summary_table,
        title="[bold green]✓ Training Completed[/bold green]",
        border_style="green",
        box=box.DOUBLE
    ))

    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

    # Restore stdout/stderr and close log file
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    _log_file.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    # Load config from YAML if specified
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Override args with config values
        for key, value in config.items():
            if value is None or value == 'null':
                setattr(args, key, None)
            elif hasattr(args, key):
                # Get the expected type from argparse default
                default_value = getattr(args, key)
                if default_value is not None:
                    expected_type = type(default_value)
                    # Keep lists as-is, convert others to expected type
                    if isinstance(value, list):
                        setattr(args, key, value)
                    else:
                        try:
                            setattr(args, key, expected_type(value))
                        except (ValueError, TypeError):
                            setattr(args, key, value)
                else:
                    setattr(args, key, value)
            else:
                setattr(args, key, value)

        console.print(f"[green]✓[/green] Loaded configuration from: [cyan]{args.config}[/cyan]")

    main(args)
