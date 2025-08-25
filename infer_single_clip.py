"""
infer_single_clip.py

Single-clip inference for SpaMo's FlanT5SLT.

Example:
  python infer_single_clip.py \
    --spatial ./features/clip-vit-large-patch14_feat_Phoenix14T/dev/myclip.npy \
    --motion  ./features/mae_feat_Phoenix14T/dev/myclip_overlap-8.npy \
    --ckpt    /abs/path/to/spamo.ckpt \
    --config  configs/finetune.yaml \
    --device  cuda:0 \
    --lang    English \
    --hf_cache_dir /data3/models

Notes:
- This script mimics the model's evaluation pipeline (prepare_visual_inputs -> prepare_inputs -> t5.generate).
- If T5 is too large to fit on your GPU, use --device cpu (slow).
"""

import argparse
import os
import sys
import numpy as np
import torch
from omegaconf import OmegaConf
import warnings
import time

# Ensure repo root is on PYTHONPATH so spamo.t5_slt imports work when running from repo
sys.path.append(os.getcwd())

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial", required=True, help="Path to spatial .npy (frames x feat_dim or 1 x frames x feat_dim)")
    p.add_argument("--motion", required=True, help="Path to motion .npy (windows x feat_dim or 1 x windows x feat_dim)")
    p.add_argument("--ckpt", required=True, help="Path to SpaMo checkpoint (.ckpt)")
    p.add_argument("--config", default="configs/finetune.yaml", help="Path to finetune.yaml (used to load model.params)")
    p.add_argument("--device", default="cpu", help="Device, e.g. cpu or cuda:0")
    p.add_argument("--lang", default="English", help="Target language used in the prompt (e.g. English)")
    p.add_argument("--hf_model_name", default=None, help="Optional override HF model name (e.g. google/flan-t5-xl)")
    p.add_argument("--hf_cache_dir", default=None, help="Optional HF cache dir")
    p.add_argument("--max_len", type=int, default=64, help="Max tokens to generate")
    p.add_argument("--num_beams", type=int, default=5, help="num beams for generation")
    p.add_argument("--do_sample", default=True, action="store_true", help="Enable sampling in generation")
    p.add_argument("--top_p", type=float, default=0.9, help="top_p for generation (nucleus)")
    p.add_argument("--print_shapes", action="store_true", help="Print intermediate tensor shapes")
    return p.parse_args()

def load_features_npy(path, device):
    arr = np.load(path, allow_pickle=False)
    # Accept [T, D] or [1, T, D] or [B, T, D]
    if arr.ndim == 2:
        t = torch.from_numpy(arr).float().unsqueeze(0)   # [1, T, D]
    elif arr.ndim == 3:
        t = torch.from_numpy(arr).float()               # [B, T, D] expected B==1
    else:
        raise ValueError(f"Unexpected array ndim for {path}: {arr.ndim}")
    return t.to(device)

def load_model_class_and_checkpoint(ckpt_path, config_path, device, hf_model_name=None, hf_cache_dir=None):
    # Import model class
    try:
        from spamo.t5_slt import FlanT5SLT
    except Exception as e:
        raise ImportError("Could not import spamo.t5_slt.FlanT5SLT. Run from repo root and ensure spamo/t5_slt.py exists.") from e

    # gather kwargs from config.model.params if available
    model_kwargs = {}
    if config_path and os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        if "model" in cfg and "params" in cfg.model:
            model_kwargs = OmegaConf.to_container(cfg.model.params, resolve=True)
            # ensure types are python-native where needed
        else:
            print("Warning: model.params not found in config; proceeding without forwarding kwargs.")
    else:
        print("Warning: config file not found; proceeding without forwarding kwargs.")

    # allow explicit overrides
    if hf_model_name is not None:
        model_kwargs["model_name"] = hf_model_name
    if hf_cache_dir is not None:
        model_kwargs["cache_dir"] = hf_cache_dir

    # Load the LightningModule from checkpoint, forwarding model kwargs
    print("Loading checkpoint (this may download HF components if necessary)...")
    try:
        model = FlanT5SLT.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"), **model_kwargs)
    except TypeError:
        # fallback if map_location not accepted by this PL version
        model = FlanT5SLT.load_from_checkpoint(ckpt_path, **model_kwargs)
    # set device attribute expected by methods and move module
    # model.device = torch.device(device)
    model.to(model.device)
    model.eval()
    return model

def build_minimal_samples(spatial_tensor, motion_tensor, lang="English", en_text="", fr_text="", es_text="", num_in_context=0):
    """
    Build a samples dict compatible with get_inputs / prepare_visual_inputs / prepare_inputs.
    spatial_tensor: [1, T, D_spatial]
    motion_tensor:  [1, W, D_motion]
    """
    pixel_values = [spatial_tensor.squeeze(0).to(dtype=torch.float32)]
    glor_values = [motion_tensor.squeeze(0).to(dtype=torch.float32)]

    # Build ex_lang_trans like the original loader does (EN=orig, FR=orig, ES=orig)
    ex_list = []
    if num_in_context and (en_text or fr_text or es_text):
        candidates = []
        if en_text:
            candidates.append(f"{en_text}={en_text}")   # if you don't have a separate short example, use same text
        if fr_text:
            candidates.append(f"{fr_text}={en_text}")
        if es_text:
            candidates.append(f"{es_text}={en_text}")
        # take up to num_in_context entries
        ex_list = [" ".join(candidates[:num_in_context])]
    else:
        # minimal safe value expected by prepare_inputs
        ex_list = [""]

    samples = {
        "pixel_values": pixel_values,
        "glor_values": glor_values,
        "num_frames": [int(spatial_tensor.shape[1])],
        "glor_lengths": [int(motion_tensor.shape[1])],
        "lang": [lang],
        "text": [""],
        "en_text": [en_text],
        "fr_text": [fr_text],
        "es_text": [es_text],
        "ex_lang_trans": ex_list,
        "id": ["single_clip"]
    }
    return samples

def main():
    args = parse_args()

    device = args.device
    # if CUDA requested but not available, fallback to cpu with a warning
    if "cuda" in device and not torch.cuda.is_available():
        warnings.warn("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    # load features to device used for constructing sample tensors (these will be moved again later if needed)
    print("Loading features...")
    spatial = load_features_npy(args.spatial, device=torch.device("cpu"))  # load initially on CPU to avoid accidental OOM
    motion  = load_features_npy(args.motion, device=torch.device("cpu"))
    print("Spatial shape:", spatial.shape, "Motion shape:", motion.shape)

    # load model + forward model.params from config (so shapes/weights align)
    model = load_model_class_and_checkpoint(
        args.ckpt, args.config, device=device,
        hf_model_name=args.hf_model_name,
        hf_cache_dir=args.hf_cache_dir
    )

    # ensure model is on the requested device (we already moved it inside load_model_class_and_checkpoint)
    model.to(torch.device(device))
    model.eval()

    # find where the model parameters actually live and use that for inputs
    model_device = next(model.parameters()).device
    spatial = spatial.to(model_device)
    motion  = motion.to(model_device)
    if args.print_shapes:
        print("After moving to model.device:", model.device)
        print("Spatial:", spatial.shape, "Motion:", motion.shape)

    # Build minimal samples dict (batch size 1)
    samples = build_minimal_samples(spatial, motion, lang=args.lang)

    # 1) prepare visual inputs -> visual_outputs, visual_masks
    #    model.prepare_visual_inputs expects samples with keys 'pixel_values', 'glor_values', 'num_frames', 'glor_lengths'
# 1) prepare visual inputs -> visual_outputs, visual_masks
    with torch.no_grad():
        visual_outputs, visual_masks = model.prepare_visual_inputs(samples)
        if args.print_shapes:
            print("visual_outputs (pre-fusion):", visual_outputs.shape)
            print("visual_masks:", visual_masks.shape)

        # IMPORTANT: Project visual outputs into T5 hidden size before prepare_inputs
        # The repo normally does: visual_outputs = self.fusion_proj(visual_outputs)
        # so we must do the same here.
        visual_outputs = model.fusion_proj(visual_outputs)
        if args.print_shapes:
            print("visual_outputs (after fusion_proj):", visual_outputs.shape)

        # 2) prepare_inputs -> input_embeds, input_masks, output_tokens, targets
        input_embeds, input_masks, _, _ = model.prepare_inputs(visual_outputs, visual_masks, samples, split="test", batch_idx=0)
        if args.print_shapes:
            print("input_embeds.shape:", input_embeds.shape, "input_masks.shape:", input_masks.shape)


        # Move inputs to the device where the underlying HF model (t5_model) resides
        t5_device = next(model.t5_model.parameters()).device
        input_embeds = input_embeds.to(t5_device)
        input_masks  = input_masks.to(t5_device)

        # 3) generate with T5
        gen_kwargs = dict(
            inputs_embeds=input_embeds,
            attention_mask=input_masks,
            num_beams=args.num_beams,
            max_length=args.max_len,
            top_p=args.top_p,
            do_sample=args.do_sample,
        )
        # Generate
        print("Running generation... (this may take a while)")
        start = time.perf_counter()
        gen_ids = model.t5_model.generate(**gen_kwargs)
        end = time.perf_counter()
        print(f"Generation completed in {end - start:.4f} seconds.")

        # 4) decode
        preds = model.t5_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        preds = [p.strip().lower() for p in preds]

    # Print result
    print("\n=== Prediction ===")
    if preds:
        print(preds[0])
    else:
        print("<empty prediction>")
    print("==================\n")

if __name__ == "__main__":
    main()
