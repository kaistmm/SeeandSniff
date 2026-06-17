import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import numpy as np

# ── Argument parsing ──────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="IIoU evaluation (mean_per_sample or average_success)")
_parser.add_argument("--gpu",        type=str, default="0",        help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')")
_parser.add_argument("--ckpt",       type=str, required=True,      help="Path to checkpoint .pth file")
_parser.add_argument("--model_mode", type=str, default="ours",     choices=["ours", "cls", "dino"], help="Heatmap mode")
_parser.add_argument("--eval_mode",  type=str, default="mean_per_sample",
                     choices=["mean_per_sample", "average_success"],
                     help="Smell aggregation mode")
_parser.add_argument("--resolution", type=int, default=224,        help="Input image resolution (e.g. 224 or 384)")
_parser.add_argument("--config",     type=str, required=True,      help="Path to YAML config that the checkpoint was trained with (e.g. configs/SeeandSniff.yaml)")
_cli = _parser.parse_args()
_RESOLUTION = _cli.resolution

os.environ['CUDA_VISIBLE_DEVICES'] = _cli.gpu

import torch
import torch.nn.functional as F

from rich.console import Console
console = Console()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from model import VisualOdorModel, ModelConfig
from dataset import group_windows_by_ingredient
from load_data import (
    load_sensor_data,
    create_label_encoder_from_json,
    diff_data_like,
    make_sliding_window_dataset,
)
from misc import compute_param_stats, print_model_tables

from types import SimpleNamespace


def override_args_with_yaml(args, yaml_path):
    """
    Override attributes of SimpleNamespace args with values from a YAML config.
    The config file is the same one used at training time (configs/*.yaml) and
    is now the single source of truth — checkpoints no longer ship with args.json.
    """
    if yaml_path is None or yaml_path == "":
        return args
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    overridden = []
    for k, v in cfg.items():
        if hasattr(args, k):
            old_val = getattr(args, k)
            if old_val != v:
                overridden.append((k, old_val, v))
            setattr(args, k, v)
        else:
            setattr(args, k, v)

    print(f"[config] Loaded {len(cfg)} keys from {yaml_path}")
    if overridden:
        print(f"[config] {len(overridden)} values changed:")
        for k, old, new in overridden:
            print(f"  {k}: {old} -> {new}")
    return args


ROOT = os.path.abspath(".")  # SeeandSniff/
args = SimpleNamespace(
    # ========= Experiment =========
    experiment_name="SeeandSniff",
    output_dir=os.path.join(ROOT, "outputs/SeeandSniff"),
    log_dir=os.path.join(ROOT, "outputs/logs/SeeandSniff"),

    # ========= Data Paths =========
    vision_train_json=os.path.join(ROOT, "metadata/train_metadata.json"),
    vision_test_json=os.path.join(ROOT, "metadata/test_metadata.json"),
    vision_train_dir=os.path.join(ROOT, "datasets/train"),
    vision_test_dir=os.path.join(ROOT, "datasets/test"),
    smell_train_dir=os.path.join(ROOT, "datasets/SmellNet/base_data/training"),
    smell_test_dir=os.path.join(ROOT, "datasets/SmellNet/base_data/testing"),
    label_json=os.path.join(ROOT, "metadata/ingredient_labels.json"),

    # ========= Smell Preprocessing =========
    diff_periods=50,
    window_size=40,
    stride=20,
    removed_columns=[
        "Benzene", "Temperature", "Pressure",
        "Humidity", "Gas_Resistance", "Altitude"
    ],
    max_seq_len=1000,

    # ========= Vision Encoder =========
    vision_encoder="dinov3_vits16",
    vision_forward_option="cls_token",
    vision_projection_type="aligner",
    vision_freeze_backbone=True,
    vision_freeze_projection=False,

    # ========= Smell Encoder =========
    smell_forward_option="cls_token",
    smell_projection_type="aligner",
    smell_model_dim=384,
    smell_num_heads=8,
    smell_num_layers=4,
    smell_freeze_backbone=False,
    smell_freeze_projection=False,

    # ========= Common =========
    embed_dim=512,

    # ========= Loss =========
    temperature=0.07,

    # ========= System =========
    device="cuda",
)


### Load training hyperparameters from YAML config
args = override_args_with_yaml(args, _cli.config)
print(args.window_size, args.stride)

# Label encoder
le = create_label_encoder_from_json(args.label_json)
console.print(f"  [green]✓[/green] Loaded [cyan]{len(le.classes_)}[/cyan] ingredient labels")


## DATASET
from torchvision import transforms
from PIL import Image

RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD  = np.array([0.229, 0.224, 0.225])

RGB_PREPROCESS_SAFE = transforms.Compose([
    transforms.Lambda(lambda im: im.convert("RGB")),
    transforms.Resize((_RESOLUTION, _RESOLUTION)),
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN.tolist(), std=RGB_STD.tolist()),
])


def load_vision_sample(path: str, transform_rgb=RGB_PREPROCESS_SAFE, device: str = None):
    rgb = Image.open(path)
    if transform_rgb is not None:
        rgb = transform_rgb(rgb)
    if device is not None:
        rgb = rgb.to(device)
    return rgb


#### Smell Samples : dict {ingredient: [list of smell samples]}
sensor_test = load_sensor_data(
    data_path=args.smell_test_dir,
    removed_filtered_columns=args.removed_columns,
)
if args.diff_periods is not None:
    console.print(f"  [dim]Applying diff with periods={args.diff_periods}...[/dim]")
    sensor_test = diff_data_like(sensor_test, periods=args.diff_periods)

if args.window_size is None:
    max_test_len = max(max(len(df) for df in dfs) for dfs in sensor_test.values())
    window_size = min(max_test_len, args.max_seq_len)
    stride = window_size
    console.print(f"  [dim]Using full sequences (test_max={max_test_len}, window={window_size}, stride={stride})...[/dim]")
else:
    if args.stride is None:
        raise ValueError("stride must be specified when window_size is set")
    window_size = args.window_size
    stride = args.stride
    console.print(f"  [dim]Creating sliding windows (window={window_size}, stride={stride})...[/dim]")

X_test_windows, y_test_windows = make_sliding_window_dataset(
    sensor_test, le, window_size=window_size, stride=stride,
)
console.print(f"  [green]✓[/green] [TEST] Created [cyan]{len(X_test_windows)}[/cyan] windows (avg [cyan]{len(X_test_windows)/50:.2f}[/cyan] per ingredient)")
console.print(f"    Window shape: [cyan]{X_test_windows.shape}[/cyan]  [dim]# (N, window_size, channels)[/dim]")

smell_test = group_windows_by_ingredient(X_test_windows, y_test_windows, le)
total_test_samples = sum(len(tensors) for tensors in smell_test.values())
console.print(f"  [green]✓[/green] [TEST] Grouped into [cyan]{len(smell_test)}[/cyan] ingredients, [cyan]{total_test_samples}[/cyan] windows")


## MODEL
sample_ing = list(smell_test.keys())[0]
sample_tensor = smell_test[sample_ing][0]
inferred_smell_input_dim = sample_tensor.shape[-1]

if args.vision_encoder in ['clip_ViT-L-14', 'clip_ViT-B-16']:
    vision_backbone = 'clip'
    vision_model_name = args.vision_encoder.replace('clip_', '')
else:
    vision_backbone = 'dino'
    vision_model_name = args.vision_encoder


config = ModelConfig(
    vision_backbone=vision_backbone,
    vision_model_name=vision_model_name,
    vision_forward_option=args.vision_forward_option,
    vision_projection_type=args.vision_projection_type,
    vision_freeze_backbone=args.vision_freeze_backbone,
    vision_freeze_projection=args.vision_freeze_projection,
    smell_input_dim=inferred_smell_input_dim,
    smell_model_dim=args.smell_model_dim,
    smell_num_heads=args.smell_num_heads,
    smell_num_layers=args.smell_num_layers,
    smell_forward_option=args.smell_forward_option,
    smell_projection_type=args.smell_projection_type,
    smell_freeze_backbone=args.smell_freeze_backbone,
    smell_freeze_projection=args.smell_freeze_projection,
    target_embedding_dim=args.embed_dim,
    init_logit_scale=np.log(1 / args.temperature),
)

device = torch.device(args.device)
model = VisualOdorModel(config).to(device)
console.print(f"  [green]✓[/green] Model created")
stats = compute_param_stats(model)
print_model_tables(console, args, inferred_smell_input_dim, stats)


### Load Weights
ckpt = _cli.ckpt
with torch.serialization.safe_globals([argparse.Namespace]):
    ckpt_obj = torch.load(ckpt, map_location=device, weights_only=False)
state_dict = ckpt_obj["model"]
if any(k.startswith("module.") for k in state_dict.keys()):
    state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
ret = model.load_state_dict(state_dict, strict=True)
print("LOAD STATE DICT RETURN:", ret)
model.eval()


# ── IIoU helpers ─────────────────────────────────────────────────────────────
def _compute_heatmap(smell_features, vision_features, target_size=None):
    if target_size is None:
        target_size = (_RESOLUTION, _RESOLUTION)
    B, D = smell_features.shape
    B, D, H, W = vision_features.shape

    vision_tok = vision_features.reshape(B, D, H * W).permute(0, 2, 1)
    smell_tok  = smell_features.unsqueeze(1)
    vision_tok = F.normalize(vision_tok, dim=-1)
    smell_tok  = F.normalize(smell_tok,  dim=-1)

    heatmap = torch.einsum("bnd,kmd->bkn", vision_tok, smell_tok).cpu().detach().numpy()
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap.reshape(H, W)
    heatmap = np.array(Image.fromarray(heatmap).resize(target_size, Image.BILINEAR))
    return heatmap


def _compute_dino_saliency_map(vision_sample, model, target_size=None):
    if target_size is None:
        target_size = (_RESOLUTION, _RESOLUTION)
    cls_token = model.vision_encoder.backbone.forward_cls_token(vision_sample)
    spatial_tokens = model.vision_encoder.backbone.forward_spatial_tokens(vision_sample)
    saliency_map = torch.einsum("bdhw,bd->bhw", spatial_tokens, cls_token).squeeze(0)
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    saliency_map = np.array(Image.fromarray(saliency_map.cpu().numpy()).resize(target_size, Image.BILINEAR))
    return saliency_map


def single_iou(heatmap: np.ndarray, binary_mask: np.ndarray, k: int = 20) -> float:
    """Best-threshold IoU between a flat heatmap and a flat binary mask."""
    pred = torch.tensor(heatmap.flatten(), dtype=torch.float32)
    tgt  = torch.tensor(binary_mask.flatten(), dtype=torch.float32)
    tgt  = tgt > 0.5

    thresholds = torch.linspace(pred.min(), pred.max(), k)
    hard_pred  = pred.unsqueeze(0) > thresholds.unsqueeze(1)
    tgt_exp    = tgt.unsqueeze(0).expand_as(hard_pred)

    intersection = torch.logical_and(hard_pred, tgt_exp).sum(dim=1).float()
    union        = torch.logical_or(hard_pred, tgt_exp).sum(dim=1).float()
    union        = torch.where(union == 0, torch.tensor(1.0), union)
    iou_scores   = intersection / union
    return torch.max(iou_scores).item()


def _mean_window_iou(vision_feat, smell_feats, binary_mask):
    """Average single_iou over all smell windows."""
    ious = []
    for w in range(smell_feats.shape[0]):
        heatmap = _compute_heatmap(smell_feats[w:w + 1], vision_feat)
        ious.append(single_iou(heatmap, binary_mask))
    return float(np.mean(ious))


def _avg_success_window_rate(vision_feat, smell_feats, binary_mask):
    """Fraction of windows where single_iou > 0.5."""
    successes = []
    for w in range(smell_feats.shape[0]):
        heatmap = _compute_heatmap(smell_feats[w:w + 1], vision_feat)
        successes.append(float(single_iou(heatmap, binary_mask) > 0.5))
    return float(np.mean(successes))


# ── Output log path ──────────────────────────────────────────────────────────
_config_stem = Path(_cli.config).stem
_ckpt_stem   = Path(_cli.ckpt).stem.replace("checkpoint_", "")
_exp_tag     = f"{_config_stem}_{_ckpt_stem}_{_cli.model_mode}_{_cli.eval_mode}_{_RESOLUTION}px"
_result_log_path = Path(__file__).parent / "results" / "IIoU" / f"{_exp_tag}.txt"
_result_log_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Results will be logged to: {_result_log_path}")


# ══════════════════════════════════════════════════════════════════════════════
# IIoU (Interactive IoU) Evaluation
# ──────────────────────────────────────────────────────────────────────────────
# Per-sample scoring:
#   eval_mode == "mean_per_sample":
#     iou1 = mean over windows of single_iou(heatmap_with_ing1_smell, mask1)
#     iou2 = mean over windows of single_iou(heatmap_with_ing2_smell, mask2)
#     success if iou1 > 0.5 AND iou2 > 0.5
#     → IIoU = succ / total
#   eval_mode == "average_success":
#     rate1 = fraction of windows where single_iou(heatmap_with_ing1_smell, mask1) > 0.5
#     rate2 = fraction of windows where single_iou(heatmap_with_ing2_smell, mask2) > 0.5
#     sample_score = rate1 × rate2
#     → AvgSucc-IIoU = mean(sample_score)
# ══════════════════════════════════════════════════════════════════════════════

# Release IIoU layout (built by build_release_dataset.py --build-iiou):
#   datasets/test_iiou_metadata.json  (per-sample paths + ingredient pair)
#   datasets/test_iiou/...            (images, 224×224)
#   datasets/mask_iiou_first/...      (mask for ingredient1)
#   datasets/mask_iiou_second/...     (mask for ingredient2)
_datasets_root      = os.path.dirname(args.vision_test_dir)        # = "<ROOT>/datasets"
_iiou_metadata_path = os.path.join(ROOT, "metadata", "test_iiou_metadata.json")
with open(_iiou_metadata_path, "r") as _f:
    _iiou_entries = json.load(_f)
# Defensive: drop entries with missing ingredient2 (shouldn't happen)
_iiou_entries = [e for e in _iiou_entries if e.get("ingredient2")]

console.print(f"\n[bold cyan]── IIoU Evaluation ({_cli.eval_mode}) ─────────────────────[/bold cyan]")
print(f"Total IIoU samples: {len(_iiou_entries)}")

model_mode = _cli.model_mode
eval_mode  = _cli.eval_mode

succ  = 0
_avg_succ_total = 0.0
total = 0
_iiou_detail = []   # [(rel_path, ing1, ing2, val1, val2, score_or_success)]

for _entry in _iiou_entries:
    _rel_path  = _entry["path"]           # e.g. "test_iiou/Fruits/Apple/iiou_001.jpg"
    _mask1_rel = _entry["mask_first"]
    _mask2_rel = _entry["mask_second"]
    _ing1      = _entry["ingredient1"]
    _ing2      = _entry["ingredient2"]

    if _ing1 not in smell_test or _ing2 not in smell_test:
        console.print(f"  [yellow][Skip][/yellow] Missing smell for '{_ing1}' or '{_ing2}' → {_rel_path}")
        continue

    _img_path   = os.path.join(_datasets_root, _rel_path)
    _mask1_path = os.path.join(_datasets_root, _mask1_rel)
    _mask2_path = os.path.join(_datasets_root, _mask2_rel)

    if not os.path.isfile(_img_path):
        console.print(f"  [yellow][Skip][/yellow] Image not found: {_img_path}")
        continue
    if not os.path.isfile(_mask1_path) or not os.path.isfile(_mask2_path):
        console.print(f"  [yellow][Skip][/yellow] Mask not found for: {_rel_path}")
        continue

    _vis_sample = load_vision_sample(_img_path, RGB_PREPROCESS_SAFE, device=device).unsqueeze(0)

    with torch.no_grad():
        if model_mode == "cls":
            if vision_backbone == "clip":
                _vis_feat = model.vision_encoder.backbone.forward_spatial(_vis_sample)
            else:
                _vis_feat = model.vision_encoder.backbone.forward_spatial_tokens(_vis_sample)
            _vis_feat = model.vision_encoder._apply_projection(_vis_feat)
            _vis_feat = F.normalize(_vis_feat, dim=1)
        else:
            _vis_feat = model.get_vision_features(_vis_sample)

    _bin_mask1 = (np.array(Image.open(_mask1_path).convert("L").resize((_RESOLUTION, _RESOLUTION), Image.NEAREST)) > 128).astype(np.uint8)
    _bin_mask2 = (np.array(Image.open(_mask2_path).convert("L").resize((_RESOLUTION, _RESOLUTION), Image.NEAREST)) > 128).astype(np.uint8)

    _smell1 = torch.stack(smell_test[_ing1]).to(device)
    _smell2 = torch.stack(smell_test[_ing2]).to(device)
    with torch.no_grad():
        _sfeat1 = model.get_smell_features(_smell1)
        _sfeat2 = model.get_smell_features(_smell2)

    if model_mode == "dino":
        # DINO saliency is vision-only and identical for both ingredients
        _hmap = _compute_dino_saliency_map(_vis_sample, model)
        if eval_mode == "average_success":
            # single heatmap → rate is 0 or 1
            _iou1 = float(single_iou(_hmap, _bin_mask1) > 0.5)
            _iou2 = float(single_iou(_hmap, _bin_mask2) > 0.5)
        else:  # mean_per_sample
            _iou1 = single_iou(_hmap, _bin_mask1)
            _iou2 = single_iou(_hmap, _bin_mask2)
    elif eval_mode == "average_success":
        _iou1 = _avg_success_window_rate(_vis_feat, _sfeat1, _bin_mask1)
        _iou2 = _avg_success_window_rate(_vis_feat, _sfeat2, _bin_mask2)
    else:  # mean_per_sample
        _iou1 = _mean_window_iou(_vis_feat, _sfeat1, _bin_mask1)
        _iou2 = _mean_window_iou(_vis_feat, _sfeat2, _bin_mask2)

    total += 1
    if eval_mode == "average_success":
        _sample_score = _iou1 * _iou2
        _avg_succ_total += _sample_score
        _iiou_detail.append((_rel_path, _ing1, _ing2, _iou1, _iou2, _sample_score))
        console.print(
            f"  [{_sample_score:.3f}] {_rel_path:<60}  {_ing1}:{_iou1:.3f}  {_ing2}:{_iou2:.3f}"
        )
    else:  # mean_per_sample
        _success = (_iou1 > 0.5) and (_iou2 > 0.5)
        succ += int(_success)
        _iiou_detail.append((_rel_path, _ing1, _ing2, _iou1, _iou2, _success))
        _mark = "[green]✓[/green]" if _success else "[red]✗[/red]"
        console.print(
            f"  {_mark} {_rel_path:<60}  {_ing1}:{_iou1:.3f}  {_ing2}:{_iou2:.3f}"
        )

if eval_mode == "average_success":
    _iiou_score = _avg_succ_total / total if total > 0 else 0.0
else:
    _iiou_score = succ / total if total > 0 else 0.0


# ── Build report (print + save) ──────────────────────────────────────────────
report_lines = [
    f"ckpt      : {_cli.ckpt}",
    f"model_mode: {_cli.model_mode}",
    f"eval_mode : {_cli.eval_mode}",
    "",
    "=" * 60,
    f"  [IIoU Evaluation: {_cli.eval_mode}]",
    "=" * 60,
]
for _rel, _i1, _i2, _v1, _v2, _suc in _iiou_detail:
    if eval_mode == "average_success":
        report_lines.append(f"  [{_suc:.3f}] {_rel:<60}  {_i1}:{_v1:.3f}  {_i2}:{_v2:.3f}")
    else:
        _m = "O" if _suc else "X"
        report_lines.append(f"  [{_m}] {_rel:<60}  {_i1}:{_v1:.3f}  {_i2}:{_v2:.3f}")
report_lines.append("")
report_lines.append("=" * 60)
if eval_mode == "average_success":
    report_lines.append(f"  AvgSucc-IIoU = {_iiou_score:.4f}  (over {total} samples)")
else:
    report_lines.append(f"  IIoU = {succ}/{total} = {_iiou_score:.4f}")
report_lines.append("=" * 60)

report = "\n".join(report_lines)
print(f"\n{report}")
with open(_result_log_path, "w") as _f:
    _f.write(report + "\n")
print(f"\nResults saved to: {_result_log_path}")
