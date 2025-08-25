#!/bin/bash

python scripts/mae_extract_feature.py \
    --anno_root ./preprocess/Phoenix14T \
    --model_name MCG-NJU/videomae-large \
    --video_root ./Phoenix \
    --cache_dir ./cache/models \
    --save_dir ./features \
    --overlap_size 8 \
    --batch_size 4 \
    --device cuda:0