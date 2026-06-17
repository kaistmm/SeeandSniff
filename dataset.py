import torch
from torch.utils.data import Dataset, Sampler
import random
import numpy as np
from collections import defaultdict


class VisualOdorDataset(Dataset):
    """
    Cross-modal Vision-Smell paired dataset.
    
    Args:
        vision_data: dict[ingredient, list[Tensor(3,224,224)]]
        smell_data: dict[ingredient, list[Tensor(T,C)]]
        le: LabelEncoder from load_gcms_data()
        pairing_mode: 
            'naive_random' - Re-pair randomly each epoch
            'fixed_random' - Fixed 1:1 pairing at init
            'cycled_shuffle' - Cycle through shuffled images
        vision_transform: optional augmentation
        smell_transform: optional augmentation
        seed: random seed
    """
    
    def __init__(
        self,
        vision_data: dict,
        smell_data: dict,
        le,
        pairing_mode: str = 'cycled_shuffle',
        seed: int = 42,
    ):
        self.vision_data = vision_data
        self.smell_data = smell_data
        self.le = le
        self.pairing_mode = pairing_mode
        self.seed = seed
        
        np.random.seed(seed)
        random.seed(seed)
        
        # Get common ingredients
        vision_set = set(vision_data.keys())
        smell_set = set(smell_data.keys())
        le_set = set(le.classes_)
        self.common_ingredients = sorted(vision_set & smell_set & le_set)
        
        # Mode-specific initialization
        self.current_epoch = 0
        if pairing_mode == 'cycled_shuffle':
            self._init_cycle_state()
        
        # Build initial pairs
        self.pairs = self._build_pairs()
        
        print(f"VisualOdorDataset: {len(self.common_ingredients)} ingredients, "
              f"{len(self.pairs)} pairs, mode={pairing_mode}")
    
    def _init_cycle_state(self):
        """Initialize state for cycled_shuffle mode."""
        self.shuffled_indices = {}
        self.current_position = {}
        
        for ingredient in self.common_ingredients:
            n_v = len(self.vision_data[ingredient])
            self.shuffled_indices[ingredient] = np.random.permutation(n_v)
            self.current_position[ingredient] = 0
    
    def _build_pairs(self):
        """Build pairs based on current mode and state."""
        pairs = []
        
        for ingredient in self.common_ingredients:
            v_list = self.vision_data[ingredient]
            s_list = self.smell_data[ingredient]
            label_idx = self.le.transform([ingredient])[0]
            
            n_v = len(v_list)
            n_s = len(s_list)
            
            if self.pairing_mode == 'naive_random':
                # Each smell picks random vision
                for s_idx in range(n_s):
                    v_idx = np.random.randint(n_v)
                    pairs.append((v_idx, s_idx, ingredient, label_idx))
            
            elif self.pairing_mode == 'fixed_random':
                # Sample N visions for N smells (1:1)
                if n_v >= n_s:
                    v_indices = np.random.choice(n_v, size=n_s, replace=False)
                else:
                    v_indices = np.random.choice(n_v, size=n_s, replace=True)
                
                for s_idx, v_idx in enumerate(v_indices):
                    pairs.append((v_idx, s_idx, ingredient, label_idx))
            
            elif self.pairing_mode == 'cycled_shuffle':
                # Take next N from shuffled cycle
                for s_idx in range(n_s):
                    pos = self.current_position[ingredient] % n_v
                    v_idx = self.shuffled_indices[ingredient][pos]
                    pairs.append((v_idx, s_idx, ingredient, label_idx))
                    self.current_position[ingredient] += 1
            
            else:
                raise ValueError(f"Unknown pairing_mode: '{self.pairing_mode}'")
        
        return pairs
    
    def on_epoch_start(self, epoch: int):
        """
        Call at epoch start to update pairing.
        
        - naive_random: Re-generate random pairs
        - fixed_random: No change
        - cycled_shuffle: Move to next cycle position
        """
        self.current_epoch = epoch
        
        if self.pairing_mode == 'naive_random':
            # Different seed per epoch
            np.random.seed(self.seed + epoch)
            random.seed(self.seed + epoch)
            self.pairs = self._build_pairs()
        
        elif self.pairing_mode == 'cycled_shuffle':
            # Check if need to re-shuffle
            for ingredient in self.common_ingredients:
                n_v = len(self.vision_data[ingredient])
                if self.current_position[ingredient] >= n_v:
                    # Complete cycle, re-shuffle
                    np.random.seed(self.seed + epoch)
                    self.shuffled_indices[ingredient] = np.random.permutation(n_v)
                    self.current_position[ingredient] = 0
            
            self.pairs = self._build_pairs()
        
        # fixed_random: do nothing
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        v_idx, s_idx, ingredient, label_idx = self.pairs[idx]
        
        vision_tensor = self.vision_data[ingredient][v_idx].clone()
        smell_tensor = self.smell_data[ingredient][s_idx].clone()
        
        return vision_tensor, smell_tensor, label_idx
    
    def get_all_labels(self):
        """Return all ingredient labels for the current pairs (for loss computation)."""
        return torch.tensor([label_idx for _, _, _, label_idx in self.pairs], dtype=torch.long)
    
    def get_stats(self):
        """Return per-ingredient statistics."""
        stats = defaultdict(lambda: {'vision': 0, 'smell': 0, 'pairs': 0})
        
        for _, _, ingredient, _ in self.pairs:
            stats[ingredient]['pairs'] += 1
        
        for ingredient in self.common_ingredients:
            stats[ingredient]['vision'] = len(self.vision_data[ingredient])
            stats[ingredient]['smell'] = len(self.smell_data[ingredient])
        
        return dict(stats)


class BalancedIngredientSampler(Sampler):
    """Round-robin sampler for diverse batches."""
    
    def __init__(self, dataset: VisualOdorDataset, batch_size: int, shuffle: bool = True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # save label information of each index from the dataset pairs
        self.label_to_indices = defaultdict(list)
        for idx, (_, _, _, label_idx) in enumerate(dataset.pairs):
            self.label_to_indices[label_idx].append(idx)
        
        if self.shuffle:
            for indices in self.label_to_indices.values():
                random.shuffle(indices)
        
        self.unique_labels = list(self.label_to_indices.keys())
    
    def __iter__(self):
        queues = {label: list(indices) for label, indices in self.label_to_indices.items()}
        all_batches = []
        
        while any(queues.values()):
            if self.shuffle:
                random.shuffle(self.unique_labels)
            
            current_batch = []
            for label in self.unique_labels:
                if queues[label]:
                    current_batch.append(queues[label].pop())
                    if len(current_batch) == self.batch_size:
                        all_batches.append(current_batch)
                        current_batch = []
            
            if current_batch:
                all_batches.append(current_batch)
        
        all_indices = [idx for batch in all_batches for idx in batch]
        return iter(all_indices)
    
    def __len__(self):
        return len(self.dataset)


def group_windows_by_ingredient(X, y, le):
    """
    Convert make_sliding_window_dataset output to dict format.
    
    Args:
        X: np.ndarray (N, T, C)
        y: np.ndarray (N,)
        le: LabelEncoder
    
    Returns:
        dict[ingredient, list[Tensor]]
    """
    result = defaultdict(list)
    
    for i in range(len(X)):
        label_idx = y[i]
        ingredient = le.inverse_transform([label_idx])[0]
        window_tensor = torch.tensor(X[i], dtype=torch.float32)
        result[ingredient].append(window_tensor)
    
    return dict(result)