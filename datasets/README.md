# Dataset Preparation
## 1. Download

Follow the steps below to obtain each dataset.

### 1.1 SmellNet Dataset
Download SmellNet from [here](https://huggingface.co/datasets/DeweiFeng/SmellNet). Only the `base_data` folder is needed; place it under `datasets/SmellNet/`.

### 1.2 SmellNet-V Dataset

1. **Download web-crawled images**

    Using `train_urls.json` and `test_urls.json`, download each image from the URL (right column) to the designated path (left column).

2. **Download evaluation masks**

    Download the mask zip files from [here](https://huggingface.co/sswwoo/SeeandSniff/tree/main/eval_masks) and extract them into `datasets/mask_test/`.

### 1.3 SmellNet-V-InteractiveSource Dataset

1. **Download web-crawled images (raw)**

    Using `test_iiou_urls.json`, download each image from the URL (right column) to the designated path (left column).

2. **Preprocess**

    Run `python datasets/preprocess_iiou.py` to apply the crop/resize recipe and produce the aligned 224x224 images at `datasets/test_iiou/`.

3. **Download evaluation masks**

    Each IIoU sample has two masks (one per ingredient). Download the mask zip files from [here](https://huggingface.co/sswwoo/SeeandSniff/tree/main/eval_masks) and extract them into `datasets/mask_iiou_first/` and `datasets/mask_iiou_second/`.


## 2. Data Structure
Expected `datasets/` layout:

```
datasets/
│   # SmellNet
├── SmellNet/
│   └── base_data/
│       ├── training/         # Training smell readings
│       └── testing/          # Test smell readings
│
│   # SmellNet-V
├── train/                    # Training images
├── test/                     # Test images
├── mask_test/                # GT masks for test images
│
│   # SmellNet-V-InteractiveSource
├── test_iiou_raw/            # User-downloaded raw images (any extension)
├── test_iiou/                # 224x224 outputs from preprocess_iiou.py
├── mask_iiou_first/          # Mask for ingredient1
└── mask_iiou_second/         # Mask for ingredient2
```
