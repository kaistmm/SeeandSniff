#!/bin/bash

# Linear Probing for Smell Encoder
# Usage: bash scripts/linear_probing.sh

# Set paths
CHECKPOINT_NAME="SeeandSniff"
CHECKPOINT_PATH="outputs/${CHECKPOINT_NAME}.pth"
CONFIG="configs/${CHECKPOINT_NAME}.yaml"
GPU_ID=0

# Run linear probing
# Note: All training configuration (model settings, preprocessing parameters) 
#       will be automatically loaded from config YAML.
CUDA_VISIBLE_DEVICES=${GPU_ID} python evaluation/classification/linear_probing_smell.py \
    --config ${CONFIG} \
    --checkpoint ${CHECKPOINT_PATH} \
    --smell_train_dir datasets/SmellNet/base_data/training \
    --smell_test_dir datasets/SmellNet/base_data/testing \
    --label_json metadata/ingredient_labels.json \
    --batch_size 64 \
    --epochs 50 \
    --lr 0.001 \
    --weight_decay 1e-4 \
    --optimizer adam \
    --num_workers 4 \
    --output_dir outputs/linear_probing/${CHECKPOINT_NAME}
