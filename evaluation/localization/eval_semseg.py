import argparse
import json
import os
import sys
from pathlib import Path

import yaml
import numpy as np

# ── Argument parsing ──────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(description="Semantic segmentation evaluation")
_parser.add_argument("--gpu",        type=str, default="0",        help="CUDA_VISIBLE_DEVICES (e.g. '0' or '0,1')")
_parser.add_argument("--ckpt",       type=str, required=True,      help="Path to checkpoint .pth file")
_parser.add_argument("--model_mode", type=str, default="ours",     choices=["ours", "cls", "dino"], help="Heatmap mode")
_parser.add_argument("--resolution", type=int, default=224,        help="Input image resolution (e.g. 224 or 384)")
_parser.add_argument("--config",     type=str, required=True,      help="Path to YAML config that the checkpoint was trained with (e.g. configs/SeeandSniff.yaml)")
_cli = _parser.parse_args()
_RESOLUTION = _cli.resolution

os.environ['CUDA_VISIBLE_DEVICES'] = _cli.gpu

import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import binary_average_precision

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


##### Vision Samples : dict {ingredient: [list of vision samples]}
test_vision_root = args.vision_test_dir
test_metadata_path = args.vision_test_json
test_vision_json = json.load(open(test_metadata_path, "r"))
test_vision_json = {ing: obj["images"] for cat, d in test_vision_json.items() for ing, obj in d.items()}
test_vision_json = {k: [os.path.join(test_vision_root, p) for p in v] for k, v in test_vision_json.items()}

vision_test = {ing: [load_vision_sample(p, RGB_PREPROCESS_SAFE, device="cpu") for p in paths]
               for ing, paths in test_vision_json.items()}

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


total_pairs = sum(len(vision_test[ing]) * len(smell_test[ing]) for ing in vision_test.keys())
print(f"Total pairs for evaluation: {total_pairs}  ( ~ 50 ingredients × ~20 vision samples × W smell samples )")


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


def multi_iou(prediction, target, k=20):
    prediction = torch.as_tensor(prediction).detach().clone()
    target = torch.as_tensor(target).detach().clone()
    target = target > 0.5

    thresholds = torch.linspace(prediction.min(), prediction.max(), k)
    hard_pred  = prediction.unsqueeze(0) > thresholds.reshape(k, 1, 1, 1, 1)
    target     = torch.broadcast_to(target.unsqueeze(0), hard_pred.shape)

    intersection = torch.logical_and(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union        = torch.logical_or(hard_pred, target).sum(dim=(1, 2, 3, 4)).float()
    union        = torch.where(union == 0, torch.tensor(1.0), union)
    iou_scores   = intersection / union
    best_iou, _  = torch.max(iou_scores, dim=0)
    return best_iou


### Build mask paths by swapping the vision_test_dir prefix and the .jpg suffix —
### never str.replace("test", ...) on the whole path, since 'test' also appears in
### the new image filenames (e.g. test_000001.jpg).
_mask_root = os.path.join(os.path.dirname(args.vision_test_dir), "mask_test")


def _vision_to_mask_path(vp: str) -> str:
    rel = os.path.relpath(vp, args.vision_test_dir)
    return os.path.join(_mask_root, rel[:-4] + ".png")  # .jpg → .png


test_mask_json = {k: [_vision_to_mask_path(v) for v in paths] for k, paths in test_vision_json.items()}

for ing, paths in test_mask_json.items():
    for p in paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Mask file not found: {p}")
console.print(f"  [green]✓[/green] All mask files exist")

assert set(test_vision_json.keys()) == set([str(k) for k in list(smell_test.keys())]), \
    "Ingredient keys in vision and smell data do not match"


# ── Output log path ──────────────────────────────────────────────────────────
# Tag = {config stem}_{epoch suffix}_{model_mode}_{resolution}px so different
# experiments (each with their own configs/*.yaml) don't collide even when their
# checkpoint filename is the standard "checkpoint_epoch_XXX.pth".
_config_stem = Path(_cli.config).stem                          # e.g. "SeeandSniff"
_ckpt_stem   = Path(_cli.ckpt).stem.replace("checkpoint_", "")  # e.g. "epoch_200"
_exp_tag     = f"{_config_stem}_{_ckpt_stem}_{_cli.model_mode}_{_RESOLUTION}px"
_result_log_path = Path(__file__).parent / "results" / f"{_exp_tag}.txt"
_result_log_path.parent.mkdir(parents=True, exist_ok=True)
print(f"Results will be logged to: {_result_log_path}")


ingredient_to_category = {
    # Nuts
    "peanuts": "Nuts", "cashew": "Nuts", "chestnuts": "Nuts", "pistachios": "Nuts",
    "almond": "Nuts", "hazelnut": "Nuts", "walnuts": "Nuts", "pecans": "Nuts",
    "brazil_nut": "Nuts", "pili_nut": "Nuts",
    # Spices
    "cumin": "Spices", "star_anise": "Spices", "nutmeg": "Spices", "cloves": "Spices",
    "ginger": "Spices", "allspice": "Spices", "chervil": "Spices", "mustard": "Spices",
    "cinnamon": "Spices", "saffron": "Spices",
    # Herbs
    "angelica": "Herbs", "garlic": "Herbs", "chives": "Herbs", "turnip": "Herbs",
    "dill": "Herbs", "mugwort": "Herbs", "chamomile": "Herbs", "coriander": "Herbs",
    "oregano": "Herbs", "mint": "Herbs",
    # Fruits
    "kiwi": "Fruits", "pineapple": "Fruits", "banana": "Fruits", "lemon": "Fruits",
    "mandarin_orange": "Fruits", "strawberry": "Fruits", "apple": "Fruits", "mango": "Fruits",
    "peach": "Fruits", "pear": "Fruits",
    # Vegetables
    "cauliflower": "Vegetables", "brussel_sprouts": "Vegetables", "broccoli": "Vegetables",
    "sweet_potato": "Vegetables", "asparagus": "Vegetables", "avocado": "Vegetables",
    "radish": "Vegetables", "tomato": "Vegetables", "potato": "Vegetables", "cabbage": "Vegetables",
}

model_mode = _cli.model_mode


# ── Per-ingredient evaluation ────────────────────────────────────────────────
results = {}  # {ingredient: {"ap": float, "iou": float}}
for ingredient in vision_test.keys():
    if ingredient not in smell_test:
        print(f"[Skip] {ingredient}: no smell data")
        continue

    _all_heatmaps = []
    _all_masks    = []

    smell_samples = torch.stack(smell_test[ingredient]).to(device)
    smell_feat    = model.get_smell_features(smell_samples)

    for i in range(len(vision_test[ingredient])):
        vision_sample = vision_test[ingredient][i].unsqueeze(0).to(device)

        mask_path = test_mask_json[ingredient][i]
        mask_img  = Image.open(mask_path).convert("L")
        binary_mask = (np.array(mask_img) > 128).astype(np.uint8)

        with torch.no_grad():
            if model_mode == "cls":
                if vision_backbone == "clip":
                    spatial_tokens = model.vision_encoder.backbone.forward_spatial(vision_sample)
                else:
                    spatial_tokens = model.vision_encoder.backbone.forward_spatial_tokens(vision_sample)
                vision_feat = model.vision_encoder._apply_projection(spatial_tokens)
                vision_feat = F.normalize(vision_feat, dim=1)
            else:
                vision_feat = model.get_vision_features(vision_sample)

            for w in range(smell_feat.shape[0]):
                if model_mode == "dino":
                    heatmap = _compute_dino_saliency_map(vision_sample, model)
                else:
                    heatmap = _compute_heatmap(smell_feat[w:w + 1], vision_feat)
                _all_heatmaps.append(heatmap.flatten())
                _all_masks.append(binary_mask.flatten())

    _heatmaps_t = torch.tensor(np.concatenate(_all_heatmaps))
    _masks_t    = torch.tensor(np.concatenate(_all_masks))
    ap  = binary_average_precision(_heatmaps_t, _masks_t).item()
    iou = multi_iou(_heatmaps_t, _masks_t).item()
    results[ingredient] = {"ap": ap, "iou": iou}


# ── Aggregate by category ────────────────────────────────────────────────────
from collections import defaultdict
category_results = defaultdict(lambda: {"ap": [], "iou": []})
for ing, scores in results.items():
    cat = ingredient_to_category.get(ing, "Unknown")
    category_results[cat]["ap"].append(scores["ap"])
    category_results[cat]["iou"].append(scores["iou"])

CATEGORIES = ["Nuts", "Spices", "Herbs", "Fruits", "Vegetables"]


all_aps  = [v["ap"]  for v in results.values()]
all_ious = [v["iou"] for v in results.values()]

# Build the full report once; print it and also write it to the log file.
report_lines = [
    f"ckpt      : {_cli.ckpt}",
    f"model_mode: {_cli.model_mode}",
    "",
]
for cat in CATEGORIES:
    cat_ious = category_results[cat]["iou"]
    cat_aps  = category_results[cat]["ap"]
    cat_map  = float(np.mean(cat_aps))  if cat_aps  else float("nan")
    cat_miou = float(np.mean(cat_ious)) if cat_ious else float("nan")
    report_lines.append("=" * 60)
    report_lines.append(f"  [{cat}]  mAP: {cat_map:.4f}  |  mIoU: {cat_miou:.4f}")
    report_lines.append("=" * 60)
    for ing in ingredient_to_category:
        if ingredient_to_category[ing] != cat:
            continue
        if ing not in results:
            report_lines.append(f"    {ing:<25}  AP: ------  IoU: ------")
        else:
            report_lines.append(f"    {ing:<25}  AP: {results[ing]['ap']:.4f}  IoU: {results[ing]['iou']:.4f}")

report_lines.append("")
report_lines.append("=" * 60)
report_lines.append(f"  [Overall]  mAP: {np.mean(all_aps):.4f}  |  mIoU: {np.mean(all_ious):.4f}")
report_lines.append(f"  (computed over {len(results)} ingredients)")
report_lines.append("=" * 60)

report = "\n".join(report_lines)
print("\n" + report)
with open(_result_log_path, "w") as _f:
    _f.write(report + "\n")
print(f"\nResults saved to: {_result_log_path}")
