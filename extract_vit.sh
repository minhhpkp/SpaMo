#!/bin/bash

python scripts/vit_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name openai/clip-vit-large-patch14 \
    --video_root ./Phoenix \
    --cache_dir ./cache/models \
    --save_dir ./features \
    --s2_mode s2wrapping \
    --scales 1 2 \
    --batch_size 14 \
    --device cuda:0