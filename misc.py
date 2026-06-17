"""
Miscellaneous utility functions for Visual-Odor training.

Contains helper functions for:
- Seed management
- Checkpoint saving/loading
"""

import os
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from rich.table import Table
from rich import box


def setup_seed(seed):
    """
    Fix random seeds for reproducibility.

    Seeds Python `random`, NumPy, and PyTorch (CPU + CUDA), and forces cuDNN
    into deterministic mode (benchmark off). Without this combination,
    cuDNN's runtime kernel auto-tuning and unseeded `random` consumers
    (e.g. dataset shufflers) drift across runs.

    Args:
        seed: Random seed
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def save_checkpoint(state, is_best, output_dir, filename='checkpoint.pth'):
    """
    Save checkpoint to disk.

    Args:
        state: Dict containing model/optimizer/scaler states
        is_best: Whether this is the best checkpoint
        output_dir: Directory to save checkpoint
        filename: Checkpoint filename
    """
    filepath = os.path.join(output_dir, filename)
    torch.save(state, filepath)
    print(f"  ✓ Saved checkpoint to {filename}")

    if is_best:
        best_filepath = os.path.join(output_dir, 'checkpoint_best.pth')
        torch.save(state, best_filepath)
        print(f"  ✓ Saved best checkpoint")


def load_checkpoint(model, optimizer, scaler, checkpoint_path, device):
    """
    Load checkpoint from disk.

    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        scaler: GradScaler to load state into
        checkpoint_path: Path to checkpoint file
        device: Device to map checkpoint to

    Returns:
        start_epoch: Epoch to resume from
        best_loss: Best loss from checkpoint
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"  Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])

    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint.get('best_loss', float('inf'))

    print(f"  ✓ Loaded checkpoint (epoch {checkpoint['epoch']}, best_loss {best_loss:.4f})")

    return start_epoch, best_loss


def load_smellnet_checkpoint(smell_encoder, checkpoint_path, device, strict=False):
    """
    Load pretrained SmellNet checkpoint into smell encoder.

    Args:
        smell_encoder: SmellEncoder module
        checkpoint_path: Path to SmellNet checkpoint (.pt file)
        device: Device to map checkpoint to
        strict: Whether to strictly match all keys

    Returns:
        checkpoint_info: Dict with model_config and spec info
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SmellNet checkpoint not found: {checkpoint_path}")

    print(f"  Loading SmellNet checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract config info
    model_config = checkpoint.get('model_config', {})
    spec = checkpoint.get('spec', {})

    # Validate checkpoint format
    if 'model_state_dict' not in checkpoint:
        raise ValueError("Invalid SmellNet checkpoint: missing 'model_state_dict'")

    state_dict = checkpoint['model_state_dict']

    # Map SmellNet keys to Visual-odor keys
    # SmellNet: input_proj, pos, transformer, dropout, classifier
    # Visual-odor: backbone (input_proj, pos, transformer, dropout), projection
    smell_state = {}
    skipped_keys = []

    for key, value in state_dict.items():
        # Skip classifier layers (we don't need them for feature extraction)
        if 'classifier' in key:
            skipped_keys.append(key)
            continue

        # Keys are already compatible - just load directly into backbone
        smell_state[key] = value

    # Load into backbone only
    missing_keys, unexpected_keys = smell_encoder.backbone.load_state_dict(smell_state, strict=strict)

    print(f"  ✓ Loaded SmellNet pretrained weights into smell encoder backbone")
    print(f"    Config: dim={model_config.get('tf_dim', 'N/A')}, "
          f"layers={model_config.get('tf_layers', 'N/A')}, "
          f"heads={model_config.get('tf_heads', 'N/A')}")
    print(f"    Gradient: {spec.get('gradient', 'N/A')}, "
          f"Window: {spec.get('window_size', 'N/A')}")

    if skipped_keys:
        print(f"    Skipped {len(skipped_keys)} classifier keys (expected)")

    if missing_keys:
        print(f"    ⚠ Missing keys: {len(missing_keys)}")
        for k in missing_keys[:3]:
            print(f"      - {k}")
        if len(missing_keys) > 3:
            print(f"      ... and {len(missing_keys)-3} more")

    if unexpected_keys:
        print(f"    ⚠ Unexpected keys: {len(unexpected_keys)}")
        for k in unexpected_keys[:3]:
            print(f"      - {k}")
        if len(unexpected_keys) > 3:
            print(f"      ... and {len(unexpected_keys)-3} more")

    return {
        'model_config': model_config,
        'spec': spec,
        'label_encoder_classes': checkpoint.get('label_encoder_classes', [])
    }

# ------------------------------------------------------------
# Helper: parameter counting and pretty printing
# ------------------------------------------------------------
def compute_param_stats(model):
    """Return param existence + counts for each submodule."""
    def count_module(module):
        if module is None:
            return 0, 0
        total = 0
        trainable = 0
        for p in module.parameters():
            total += p.numel()
            if p.requires_grad:
                trainable += p.numel()
        return total, trainable

    v_backbone_p, v_backbone_t = count_module(model.vision_encoder.backbone)
    v_proj_p, v_proj_t = count_module(model.vision_encoder.projection)
    s_backbone_p, s_backbone_t = count_module(model.smell_encoder.backbone)
    s_proj_p, s_proj_t = count_module(model.smell_encoder.projection)

    logit = getattr(model, "logit_scale", None)
    logit_params = logit.numel() if logit is not None else 0
    logit_trainable = logit_params if (logit is not None and logit.requires_grad) else 0

    total_params = v_backbone_p + v_proj_p + s_backbone_p + s_proj_p + logit_params
    total_trainable = v_backbone_t + v_proj_t + s_backbone_t + s_proj_t + logit_trainable

    return {
        "vision_backbone": {"exist": model.vision_encoder.backbone is not None, "params": v_backbone_p, "trainable": v_backbone_t},
        "vision_projection": {"exist": model.vision_encoder.projection is not None, "params": v_proj_p, "trainable": v_proj_t},
        "smell_backbone": {"exist": model.smell_encoder.backbone is not None, "params": s_backbone_p, "trainable": s_backbone_t},
        "smell_projection": {"exist": model.smell_encoder.projection is not None, "params": s_proj_p, "trainable": s_proj_t},
        "logit_scale": {"exist": logit is not None, "params": logit_params, "trainable": logit_trainable},
        "total_params": total_params,
        "total_trainable": total_trainable,
    }


def print_model_tables(console, args, inferred_smell_input_dim, stats):
    """Render model config and parameter breakdown tables."""
    config_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
    config_table.add_column(style="dim")
    config_table.add_column()
    config_table.add_row("Vision encoder", f"[cyan]{args.vision_encoder}[/cyan]")
    config_table.add_row("Vision forward", f"[cyan]{args.vision_forward_option}[/cyan]")
    config_table.add_row("Smell input dim", f"[cyan]{inferred_smell_input_dim}[/cyan] [dim](inferred from data)[/dim]")
    config_table.add_row("Smell model dim", f"[cyan]{args.smell_model_dim}[/cyan]")
    config_table.add_row("Smell forward", f"[cyan]{args.smell_forward_option}[/cyan]")
    config_table.add_row("Embed dim", f"[cyan]{args.embed_dim}[/cyan]")
    console.print(config_table)

    console.print()
    param_table = Table(show_header=True, box=box.ROUNDED, padding=(0, 1))
    param_table.add_column("Model", style="bold")
    param_table.add_column("Element", style="dim")
    param_table.add_column("Exist", justify="center")
    param_table.add_column("Trainable", justify="center")
    param_table.add_column("# of Params", justify="right", style="cyan")
    param_table.add_column("# of Trainable", justify="right", style="green")

    def add_row(model_label, element, key, show_model_label=False):
        entry = stats[key]
        exist_mark = "✓" if entry["exist"] else "✗"
        train_mark = "✓" if entry["trainable"] > 0 else "✗"
        param_table.add_row(
            model_label if show_model_label else "",
            element,
            f"[green]{exist_mark}[/green]" if exist_mark == "✓" else f"[red]{exist_mark}[/red]",
            f"[green]{train_mark}[/green]" if train_mark == "✓" else f"[red]{train_mark}[/red]",
            f"{entry['params']:,}" if entry["params"] > 0 else "-",
            f"{entry['trainable']:,}" if entry["trainable"] > 0 else "0"
        )

    add_row("VISION", "backbone", "vision_backbone", show_model_label=True)
    add_row("", "aligner", "vision_projection")
    add_row("SMELL", "backbone", "smell_backbone", show_model_label=True)
    add_row("", "aligner", "smell_projection")

    # Logit scale
    entry = stats["logit_scale"]
    exist_mark = "✓" if entry["exist"] else "✗"
    train_mark = "✓" if entry["trainable"] > 0 else "✗"
    param_table.add_row(
        "LOGIT_SCALE",
        "",
        f"[green]{exist_mark}[/green]" if exist_mark == "✓" else f"[red]{exist_mark}[/red]",
        f"[green]{train_mark}[/green]" if train_mark == "✓" else f"[red]{train_mark}[/red]",
        f"{entry['params']:,}" if entry["params"] > 0 else "-",
        f"{entry['trainable']:,}" if entry["trainable"] > 0 else "0"
    )

    param_table.add_section()
    param_table.add_row(
        "[bold]TOTAL[/bold]",
        "",
        "",
        "",
        f"[bold cyan]{stats['total_params']:,}[/bold cyan]",
        f"[bold green]{stats['total_trainable']:,}[/bold green]"
    )

    console.print(param_table)
