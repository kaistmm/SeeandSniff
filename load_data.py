import os
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
from metadata.ingredient_to_category import ingredient_to_category
import re
from collections import defaultdict
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import json


# --- Centralized Label Management ---
def load_ingredient_labels(json_path='metadata/ingredient_labels.json'):
    """Load ingredient label metadata from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)


def create_label_encoder_from_json(json_path='metadata/ingredient_labels.json'):
    """
    Create a fitted LabelEncoder from centralized JSON file.
    
    This ensures consistent label encoding across dataset and loss computation.
    
    Args:
        json_path: Path to ingredient_labels.json
    
    Returns:
        LabelEncoder with classes_ already set
    """
    metadata = load_ingredient_labels(json_path)
    encoder = LabelEncoder()
    encoder.classes_ = np.array(metadata['ingredients'])
    return encoder

# --- Statstics for normalization ---
RGB_MEAN = np.array([0.485, 0.456, 0.406])
RGB_STD = np.array([0.229, 0.224, 0.225])

# -- Unnormalization transform ---
def unnormalize_fn(mean : tuple, std : tuple) -> transforms.Compose:
    """
    returns a transformation that turns torch tensor to PIL Image
    """
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=tuple(-m / s for m, s in zip(mean, std)),
                std=tuple(1.0 / s for s in std),
            ),
            transforms.Lambda(lambda x: torch.clamp(x, 0., 1.)), 
            transforms.ToPILImage(),
        ]
    )

#-------------------------------------------
#         Data Augmentations
#-------------------------------------------

### Vision Augmentations

class RandomDiscreteRotation(nn.Module):
    """Rotate by one of the given angles."""
    def __init__(self, angles):
        self.angles = angles
    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

TO_TENSOR = transforms.Compose([
    transforms.ToTensor()
])

RGB_AUGMENTS = transforms.Compose([
    # 1. Geometric (Sizing)
    transforms.RandomResizedCrop(
        (224, 224), 
        scale=(0.9, 1.0),
        ratio=(0.75, 1.33), 
        interpolation=transforms.InterpolationMode.BILINEAR
    ),
    transforms.RandomHorizontalFlip(),
    RandomDiscreteRotation([0, 90]),
    
    # # 2. Color (Shadow Problem)
    # transforms.RandomApply([transforms.ColorJitter(
    #     brightness=(0.9, 1.1),
    #     contrast=(0.9, 1.1),
    #     saturation=0.1,
    #     hue=0.1
    # )], p=.8),
    # transforms.RandomGrayscale(p=0.1),
    
    # 3. Blur
    transforms.RandomApply([transforms.GaussianBlur(9, sigma=(0.1, 0.5))], p=0.1),
    
    # 4. Normalize
    transforms.ToTensor(),
    transforms.Normalize(mean=RGB_MEAN, std=RGB_STD),
])

RGB_PREPROCESS = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=RGB_MEAN,
        std=RGB_STD,
    ),
])

### Smell Augmentations
def apply_random_feature_dropout(X, dropout_fraction=0.25, seed=None):
    """
    Apply random feature(sensor value) dropout to a batch or dataset.

    Parameters:
    - X: torch.Tensor or np.ndarray, shape [batch_size, time_steps, feature_dim] or [batch_size, feature_dim]
    - dropout_fraction: float, fraction of features to zero out (e.g., 0.25 → drop 25%)
    - seed: int or None, random seed for reproducibility

    Returns:
    - X_dropped: same type as input, with specified features zeroed out
    """
    if seed is not None:
        np.random.seed(seed)

    if isinstance(X, np.ndarray):
        X = torch.tensor(X)

    feature_dim = X.shape[-1]
    num_features_to_drop = int(feature_dim * dropout_fraction)

    # Randomly select feature indices to drop
    drop_indices = np.random.choice(feature_dim, num_features_to_drop, replace=False)
    mask = torch.ones(feature_dim)
    mask[drop_indices] = 0

    # Apply mask
    X_dropped = X * mask.to(X.device)

    return X_dropped


def apply_noise_injection(X, noise_scale=0.05, seed=None):
    """
    Add Gaussian noise to the input tensor.

    Parameters:
    - X: torch.Tensor, shape [batch_size, time_steps, feature_dim] or [batch_size, feature_dim]
    - noise_scale: float, standard deviation of Gaussian noise
    - seed: int or None, for reproducibility

    Returns:
    - X_noisy: torch.Tensor, same shape as input
    """
    if seed is not None:
        torch.manual_seed(seed)

    noise = torch.randn_like(X) * noise_scale
    X_noisy = X + noise
    return X_noisy

def diff_data_like(
    data: dict,
    periods: int = 25,
):
    if periods == 0:
        return data  # No-op: return original data unchanged

    out = {}
    for label, dfs in data.items():
        out_list = []
        for df in dfs:
            diff_df = df.diff(periods=periods).iloc[periods:]
            out_list.append(diff_df)
        out[label] = out_list
    return out

#-------------------------------------------
#           Vision Processing 
#-------------------------------------------
def load_vision_data(
        json_path: str,
        base_image_dir: str,
        ingredients = None,
        categories = None,
        transform_rgb = None,
        device: str = None,
):
    """
    Load vision data from JSON metadata file.
    
    Parameters:
    - json_path: Path to metadata JSON file (e.g., 'metadata/train_metadata.json')
    - base_image_dir: Base directory for images (e.g., 'crawling/datasets/train')
    - ingredients: List of specific ingredients to load (optional)
    - categories: List of categories to load (optional)
    - transform_rgb: Transform to apply to images (default: RGB_PREPROCESS)
    - device: Device to move tensors to (optional)
    
    Returns:
    - data: dict[ingredient_name, list[Tensor]]
    """
    if transform_rgb is None:
        transform_rgb = RGB_PREPROCESS
    
    # Load metadata
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    data = defaultdict(list)
    
    # Iterate through categories and ingredients
    for category, ingredients_dict in metadata.items():
        # Filter by categories
        if categories and category not in categories:
            continue
        
        for ingredient, ing_data in ingredients_dict.items():
            # Filter by ingredients
            if ingredients and ingredient not in ingredients:
                continue
            
            # Process images
            image_paths = ing_data.get('images', [])
            
            for rel_path in image_paths:
                img_path = os.path.join(base_image_dir, rel_path)
                
                try:
                    rgb = Image.open(img_path)
                    if transform_rgb is not None:
                        rgb = transform_rgb(rgb)
                    if device is not None:
                        rgb = rgb.to(device)
                    
                    data[ingredient].append(rgb)
                    
                except Exception as e:
                    print(f"Failed to load {img_path}: {e}")
    
    return data

#-------------------------------------------
#           Smell Processing 
#-------------------------------------------

### Load sensor data from CSV files
def load_sensor_data(
        data_path: str,
        ingredients = None,
        categories = None,
        removed_filtered_columns = [],
):
    """
    Load smell sensor data from CSV files.
    
    Parameters:
    - data_path: Path to directory containing ingredient folders with CSV files
    - ingredients: List of specific ingredients to load (optional)
    - categories: List of categories to load (optional)
    - removed_filtered_columns: List of column names to remove
    
    Returns:
    - data: dict[ingredient_name, list[DataFrame]]
    """
    data = defaultdict(list)
    
    # Helper: subtract first row (baseline correction)
    def subtract_first_row(df):
        return df - df.iloc[0]
    
    # Walk through the directory
    for folder_name in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder_name)
        
        # Filter by ingredients
        if ingredients and folder_name not in ingredients:
            continue
        
        # Filter by categories
        if categories and ingredient_to_category[folder_name] not in categories:
            continue
        
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                if file_name.endswith('.csv'):
                    cur_path = os.path.join(folder_path, file_name)
                    try:
                        df = pd.read_csv(cur_path)
                        df = subtract_first_row(df)
                        df = df.drop(columns=removed_filtered_columns, errors='ignore')
                        data[folder_name].append(df)
                    except Exception as e:
                        print(f"Failed to load {cur_path}: {e}")
    
    return data

### Load GCMS data
def load_gcms_data(path, label_json_path='metadata/ingredient_labels.json'):
    """
    Load GCMS data with centralized label encoding.
    
    Args:
        path: Path to GCMS CSV file
        label_json_path: Path to ingredient_labels.json
    
    Returns:
        X_scaled: Normalized features
        y_encoded: Encoded labels
        le: LabelEncoder (from JSON)
        scaler: StandardScaler
    """
    df = pd.read_csv(path)

    feature_cols = df.columns[1:]
    label_col = df.columns[0]

    # Extract features and labels
    X = df[feature_cols].values
    y = df[label_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use centralized label encoder
    le = create_label_encoder_from_json(label_json_path)
    
    # Validate all ingredients are in the encoder
    unknown = set(y) - set(le.classes_)
    if unknown:
        raise ValueError(f"Unknown ingredients in CSV: {unknown}")
    
    y_encoded = le.transform(y)

    return X_scaled, y_encoded, le, scaler

def make_sliding_window_dataset(
        data: dict[str, list[pd.DataFrame]],
        le,
        window_size: int = 100,
        stride: int = 50,
):
    """
    Build a windowed time-series dataset from {label: [DataFrame, ...]}.

    Returns
    -------
    X : np.ndarray, shape [N, window_size, C]
        Stacked sliding windows of features.
    y : np.ndarray, shape [N]
        Label-encoded class IDs aligned with X.
    label_encoder : same as input
        Returned unchanged; must be pre-fitted (uses .transform()).
    """
    X = [] # for windows
    y = [] # for ingredient labels

    for ingredient, dfs in data.items():
        for df in dfs:
            for start in range(0, len(df)- window_size +1, stride):
                window = df.iloc[start : start+ window_size].values
                X.append(window)
                y.append(ingredient)
    y =le.transform(y)
    X = np.array(X) # shape: [N, window_size, C]

    return X, y


#-------------------------------------------
#       Text Processing (Just in case)
#-------------------------------------------
def load_text_data(path, le=None, label_json_path='metadata/ingredient_labels.json'):
    text_embeddings = np.load(path, allow_pickle=True).item()

    X = np.array([value for _, value in text_embeddings.items()])
    y = list(text_embeddings.keys())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Encode labels
    if le is None:
        le = create_label_encoder_from_json(label_json_path)
        y_encoded = le.transform(y)
    else:
        y_encoded = le.transform(y)

    return X_scaled, y_encoded, le, scaler
