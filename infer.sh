#!/bin/bash

python infer_single_clip.py \
  --spatial ./features/clip-vit-large-patch14_feat_Phoenix14T/test/25October_2010_Monday_tagesschau-24_s2wrapping.npy \
  --motion "./features/mae_feat_Phoenix14T/test/25October_2010_Monday_tagesschau-24_overlap-8.npy" \
  --ckpt ./spamo.ckpt \
  --device cpu \
  --config configs/finetune.yaml \
  --hf_model_name google/flan-t5-xl \
  --hf_cache_dir ./cache/models \
  --lang German

# python main.py -c configs/finetune.yaml -e bleu --train False --test True --ckpt spamo.ckpt