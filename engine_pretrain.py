"""
Training and evaluation engine for Visual-Odor cross-modal learning.

Includes:
- train_one_epoch: Training loop with mixed precision, iteration-wise LR scheduling
- evaluate: Validation loop with retrieval metrics
"""

import math
import numpy as np
import torch
import torch.nn.functional as F
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table
from rich import box
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
import util.lr_sched as lr_sched

console = Console()


def train_one_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    args,
    log_writer: Optional[object] = None,
):
    """
    Train for one epoch.

    Args:
        model: Visual-Odor model
        loss_fn: VisualOdorLoss instance
        dataloader: Training dataloader
        optimizer: PyTorch optimizer
        scaler: GradScaler for mixed precision
        device: Device (cuda/cpu)
        epoch: Current epoch number (0-indexed)
        args: Arguments object with lr, min_lr, warmup_epochs, epochs
        log_writer: TensorBoard writer (optional)

    Returns:
        dict: Average metrics for the epoch
    """
    model.train()

    # Metrics tracking
    total_loss = 0.0
    num_batches = 0

    # Rich progress bar
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
    )
    progress.start()
    task = progress.add_task(
        f"[cyan]Epoch {epoch}/{args.epochs-1}",
        total=len(dataloader)
    )

    for data_iter_step, batch in enumerate(dataloader):
        # 1. Iteration-wise learning rate scheduling using fractional epoch
        lr = lr_sched.adjust_learning_rate(
            optimizer,
            data_iter_step / len(dataloader) + epoch,
            args
        )

        # 2. Data to device
        vision = batch[0].to(device, non_blocking=True)
        smell = batch[1].to(device, non_blocking=True)
        labels = batch[2]  # Keep on CPU for now

        # 3. Mixed precision forward pass
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output_dict = model(vision=vision, smell=smell)
            loss = loss_fn(output_dict, return_metrics=False)

        # 4. Check for invalid loss
        if not math.isfinite(loss.item()):
            print(f"Loss is {loss.item()}, stopping training")
            raise ValueError(f"Loss became {loss.item()}")

        # 5. Scaled backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()

        # Optional gradient clipping
        if hasattr(args, 'clip_grad') and args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        scaler.step(optimizer)
        scaler.update()

        # 6. Update metrics
        total_loss += loss.item()
        num_batches += 1

        # 7. Update progress bar
        avg_loss = total_loss / num_batches
        progress.update(
            task,
            advance=1,
            description=f"[cyan]Epoch {epoch}/{args.epochs-1} | Loss: {avg_loss:.4f} | LR: {lr:.6f}"
        )

        # 8. Log to tensorboard
        if log_writer is not None and data_iter_step % 100 == 0:
            global_step = epoch * len(dataloader) + data_iter_step
            log_writer.add_scalar('train/loss', loss.item(), global_step)
            log_writer.add_scalar('train/lr', lr, global_step)

    # Stop progress bar
    progress.stop()

    # Return epoch metrics
    metrics = {
        'loss': total_loss / num_batches,
        'lr': lr
    }

    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    vision_test: dict,
    smell_test: dict,
    le,
    device: torch.device,
    epoch: Optional[int] = None,
    log_writer: Optional[object] = None,
):
    """
    Evaluate on test set using vision_test and smell_test dictionaries directly.

    Args:
        model: Visual-Odor model
        loss_fn: VisualOdorLoss instance (with retrieval_mode set)
        vision_test: Dict mapping ingredient -> list of vision tensors
        smell_test: Dict mapping ingredient -> list of smell tensors
        le: LabelEncoder for ingredient labels
        device: Device (cuda/cpu)
        epoch: Current epoch (for logging)
        log_writer: TensorBoard writer (optional)

    Returns:
        dict: Evaluation metrics including loss and retrieval accuracies
    """
    model.eval()

    console.print(f"\n[bold cyan]→[/bold cyan] Extracting features from test set...")

    # 1. Extract vision features
    all_vision_feats = []
    all_vision_labels = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    progress.start()
    task = progress.add_task("[green]Extracting vision features", total=len(vision_test))

    for ingredient in vision_test.keys():
        label_idx = le.transform([ingredient])[0]
        for vision_tensor in vision_test[ingredient]:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                vision_feat = model.get_vision_features(
                    vision_tensor.unsqueeze(0).to(device)
                )
            all_vision_feats.append(vision_feat.float().cpu())
            all_vision_labels.append(label_idx)
        progress.update(task, advance=1)

    progress.stop()

    # 2. Extract smell features
    all_smell_feats = []
    all_smell_labels = []

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    progress.start()
    task = progress.add_task("[green]Extracting smell features", total=len(smell_test))

    for ingredient in smell_test.keys():
        label_idx = le.transform([ingredient])[0]
        for smell_tensor in smell_test[ingredient]:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                smell_feat = model.get_smell_features(
                    smell_tensor.unsqueeze(0).to(device)
                )
            all_smell_feats.append(smell_feat.float().cpu())
            all_smell_labels.append(label_idx)
        progress.update(task, advance=1)

    progress.stop()

    # 3. Concatenate features
    all_vision_feats = torch.cat(all_vision_feats, dim=0)  # [N_vision, D] or [N_vision, D, H, W]
    all_smell_feats = torch.cat(all_smell_feats, dim=0)    # [N_smell, D]
    all_vision_labels = torch.tensor(all_vision_labels, dtype=torch.long)
    all_smell_labels = torch.tensor(all_smell_labels, dtype=torch.long)

    console.print(f"  [green]✓[/green] Vision features: [cyan]{all_vision_feats.shape}[/cyan]")
    console.print(f"  [green]✓[/green] Smell features: [cyan]{all_smell_feats.shape}[/cyan]")

    # 4. Compute affinity matrix
    console.print(f"[bold cyan]→[/bold cyan] Computing affinity matrix...")

    # Get logit scale from model
    logit_scale = model.logit_scale.exp()

    # Determine if spatial or global mode
    is_spatial = (loss_fn.forward_mode == 'local' and all_vision_feats.dim() == 4)

    if is_spatial:
        # Spatial mode: vision [N_vision, D, H, W], smell [N_smell, D]
        N_vision = all_vision_feats.shape[0]
        N_smell = all_smell_feats.shape[0]

        # Compute: vision [N_vision, D, H, W] × smell [N_smell, D] → [N_vision, N_smell, H*W]
        affinity_matrix = torch.einsum(
            "ndhw,md->nmhw",
            all_vision_feats.to(device),
            all_smell_feats.to(device)
        )
        affinity_matrix = affinity_matrix.reshape(N_vision, N_smell, -1)  # [N_vision, N_smell, H*W]

        # Apply pooling (same as training)
        if loss_fn.spatial_pool == 'max':
            affinity_matrix = affinity_matrix.max(dim=-1)[0]  # [N_vision, N_smell]
        else:  # mean
            affinity_matrix = affinity_matrix.mean(dim=-1)  # [N_vision, N_smell]

        # Apply logit scale
        affinity_matrix = affinity_matrix * logit_scale

        # Transpose for smell-to-vision retrieval: [N_smell, N_vision]
        affinity_matrix = affinity_matrix.T

    else:
        # Global mode: vision [N_vision, D], smell [N_smell, D]
        vision_flat = all_vision_feats.reshape(all_vision_feats.shape[0], -1).to(device)
        smell_flat = all_smell_feats.reshape(all_smell_feats.shape[0], -1).to(device)

        # Compute: smell × vision^T → [N_smell, N_vision]
        affinity_matrix = logit_scale * (smell_flat @ vision_flat.T)

    console.print(f"  [green]✓[/green] Affinity matrix: [cyan]{affinity_matrix.shape}[/cyan]")

    # 5. Compute retrieval metrics
    console.print(f"[bold cyan]→[/bold cyan] Computing retrieval metrics...")

    # affinity_matrix is [N_smell, N_vision]
    # Transpose to [N_vision, N_smell] for compute_ingredientwise_retrieval
    # which expects: rows = query (vision), columns = retrieved (smell)
    affinity_v2s = affinity_matrix.T  # [N_vision, N_smell]

    if loss_fn.retrieval_mode == 'ingredientwise':
        retrieval_metrics = loss_fn.compute_ingredientwise_retrieval(
            affinity_v2s,
            all_vision_labels.to(device),
            all_smell_labels.to(device),
            topk=(1, 5, 10)
        )
    else:
        retrieval_metrics = loss_fn.compute_samplewise_retrieval(
            affinity_v2s,
            topk=(1, 5, 10)
        )

    # 6. Compute contrastive loss (skip for ingredientwise evaluation)
    loss = 0.0

    # 7. Print results
    console.print("\n[bold]Evaluation Results[/bold]")

    eval_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    eval_table.add_column(style="dim")
    eval_table.add_column()

    for key, value in retrieval_metrics.items():
        eval_table.add_row(key, f"[cyan]{value:.2f}%[/cyan]")

    console.print(eval_table)

    # 8. Log to tensorboard
    if log_writer is not None and epoch is not None:
        for key, value in retrieval_metrics.items():
            log_writer.add_scalar(f'val/{key}', value, epoch)

    # 9. Return metrics
    result = {'loss': loss}
    result.update(retrieval_metrics)

    return result
