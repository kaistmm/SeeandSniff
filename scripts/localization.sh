#!/bin/bash

# python evaluation/localization/eval_semseg.py  --ckpt outputs/Global-CLIP.pth --config configs/Global-CLIP.yaml --model_mode cls --gpu 0
# python evaluation/localization/eval_semseg.py  --ckpt outputs/Ours-Global.pth --config configs/Ours-Global.yaml --model_mode cls --gpu 0
# python evaluation/localization/eval_semseg.py  --ckpt outputs/Ours-Local.pth --config configs/Ours-Local.yaml --model_mode ours --gpu 0
python evaluation/localization/eval_semseg.py  --ckpt outputs/SeeandSniff.pth --config configs/SeeandSniff.yaml --model_mode ours --gpu 0
