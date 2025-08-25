# infer_single_clip.py
"""
Single-clip inference helper for SpaMo (Spatio-Motion SLT).
Usage (example):
  python infer_single_clip.py \
    --spatial ./features/clip-vit-large-patch14_feat_Phoenix14T/dev/myclip.npy \
    --motion ./features/mae_feat_Phoenix14T/dev/myclip_overlap-8.npy \
    --ckpt /abs/path/to/spamo.ckpt \
    --device cuda:0
"""

import argparse
import os
import numpy as np
import torch
import warnings
from omegaconf import OmegaConf

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--spatial", required=True)
    p.add_argument("--motion", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", default="configs/finetune.yaml",
                   help="Path to finetune.yaml (used to read model.params for loading)")
    p.add_argument("--max_tokens", type=int, default=256)
    p.add_argument("--print_verbose", action="store_true")
    # keep hf_model_name/hf_cache_dir as overrides if needed
    p.add_argument("--hf_model_name", type=str, default=None)
    p.add_argument("--hf_cache_dir", type=str, default=None)
    return p.parse_args()


def load_checkpoint(Cls, ckpt_path, device, model_kwargs=None):
    """
    model_kwargs: dict of constructor kwargs to forward into Cls.__init__
    """
    load_kwargs = {}
    if model_kwargs:
        load_kwargs.update(model_kwargs)

    try:
        model = Cls.load_from_checkpoint(ckpt_path, map_location=torch.device(device), **load_kwargs)
    except TypeError:
        model = Cls.load_from_checkpoint(ckpt_path, **load_kwargs)
        model.to(device)
    return model


def load_features(path, device):
    arr = np.load(path, allow_pickle=False)
    # ensure float32
    t = torch.from_numpy(arr).float()
    # add batch dim in front
    if t.dim() == 2:
        t = t.unsqueeze(0)   # [1, T, D]
    return t.to(device)

def try_load_model_class():
    # Import the model class from the repo
    try:
        from spamo.t5_slt import FlanT5SLT
        return FlanT5SLT
    except Exception as e:
        raise ImportError("Could not import spamo.t5_slt.FlanT5SLT from repo. "
                          "Make sure you're running this from the repo root and the module exists.") from e

def get_tokenizer(model):
    # Prefer model.tokenizer if available; else try to infer a HF model_name from hparams
    tokenizer = None
    try:
        tokenizer = getattr(model, "tokenizer", None)
    except Exception:
        tokenizer = None

    if tokenizer is not None:
        return tokenizer

    # Try to find model_name in attributes / hparams
    model_name = None
    for attr in ("model_name", "hparams", "cfg", "config"):
        val = getattr(model, attr, None)
        if isinstance(val, str) and "t5" in val:
            model_name = val
            break
        if hasattr(val, "get"):
            # e.g. hparams is dict-like
            model_name = val.get("model_name") or val.get("model", {}).get("model_name")
            if model_name:
                break

    if model_name is None:
        model_name = "google/flan-t5-xl"  # fallback

    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        warnings.warn(f"Failed to load tokenizer for {model_name}: {e}. Token decoding may fail.")
        return None

def decode_tokens(tokens, tokenizer):
    if tokenizer is None:
        return str(tokens)
    # tokens may be tensor, list[int], or nested
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().tolist()
    # if nested, pick first sequence
    if isinstance(tokens, list) and tokens and isinstance(tokens[0], list):
        tokens = tokens[0]
    try:
        return tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception:
        # some tokenizers require removal of ints > vocab; fallback to str
        return str(tokens)

def extract_prediction_from_output(out, tokenizer, print_verbose=False):
    # Cases handled:
    # - out is a str or list[str]
    # - out is torch.Tensor of token ids
    # - out is numpy array of ids
    # - out is dict containing 'preds','pred','prediction','logits', etc.
    if print_verbose:
        print("-- output type:", type(out))

    # direct string
    if isinstance(out, str):
        return out

    if isinstance(out, list) and out and isinstance(out[0], str):
        return out[0]

    if isinstance(out, torch.Tensor) and out.dtype in (torch.int64, torch.int32, torch.int16, torch.uint8):
        return decode_tokens(out, tokenizer)

    if isinstance(out, np.ndarray):
        if out.dtype.kind in ("i","u"):
            return decode_tokens(out, tokenizer)
        else:
            # likely floats/logits
            pass

    if isinstance(out, dict):
        # common keys to inspect
        for k in ("preds", "pred", "pred_tokens", "pred_ids", "predictions", "outputs", "logits"):
            if k in out:
                cand = out[k]
                # if logits, take argmax
                if isinstance(cand, torch.Tensor) and cand.dtype.is_floating_point:
                    ids = torch.argmax(cand, dim=-1)
                    return decode_tokens(ids, tokenizer)
                return extract_prediction_from_output(cand, tokenizer, print_verbose=print_verbose)

    # If it's a tensor of floats (e.g. logits), argmax along last dim
    if isinstance(out, torch.Tensor) and out.dtype.is_floating_point:
        ids = torch.argmax(out, dim=-1)
        return decode_tokens(ids, tokenizer)

    # If it's a list of ints
    if isinstance(out, list) and out and isinstance(out[0], int):
        return decode_tokens(out, tokenizer)

    # Fallback: convert to string
    try:
        return str(out)
    except Exception:
        return "<unprintable output>"

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # load features
    if not os.path.exists(args.spatial):
        raise FileNotFoundError("Spatial .npy not found: " + args.spatial)
    if not os.path.exists(args.motion):
        raise FileNotFoundError("Motion .npy not found: " + args.motion)

    spatial = load_features(args.spatial, device)
    motion = load_features(args.motion, device)

    print("Loaded spatial:", spatial.shape, "motion:", motion.shape)

    # import model class and load
    Cls = try_load_model_class()
    print("Loaded model class:", Cls)
    print("Loading checkpoint (this may print a lot)...")

    # load model config from YAML if provided
    model_kwargs = {}
    if args.config and os.path.exists(args.config):
        cfg = OmegaConf.load(args.config)
        # extract model.params if present
        if "model" in cfg and "params" in cfg.model:
            model_kwargs = OmegaConf.to_container(cfg.model.params, resolve=True)
        else:
            print("Warning: model.params not found in config; loading without extra kwargs")

    # allow CLI overrides for hf_model_name/hf_cache_dir
    if args.hf_model_name:
        model_kwargs["model_name"] = args.hf_model_name
    if args.hf_cache_dir:
        model_kwargs["cache_dir"] = args.hf_cache_dir

    # load checkpoint with full kwargs
    model = load_checkpoint(Cls, args.ckpt, device=device, model_kwargs=model_kwargs)

    model.eval()
    model.to(device)
    print("Model loaded. Device:", next(model.parameters()).device)
    

    # get tokenizer
    tokenizer = get_tokenizer(model)
    if tokenizer is not None:
        print("Tokenizer ready.")
    else:
        print("No tokenizer available; outputs will be printed as raw ids/strings.")

    # Try multiple common inference entry points
    out = None
    tried = []
    with torch.no_grad():
        # 1) if module has translate()
        if hasattr(model, "translate"):
            tried.append("model.translate(spatial, motion)")
            try:
                out = model.translate(spatial, motion)
            except Exception as e:
                warnings.warn(f"model.translate failed: {e}")

        # 2) if module has generate()
        if out is None and hasattr(model, "generate"):
            tried.append("model.generate(spatial, motion)")
            try:
                # many custom modules wrap HF generate differently; try simple calls
                out = model.generate(spatial, motion)
            except Exception as e:
                warnings.warn(f"model.generate(spatial,motion) failed: {e}")
                # try named args
                try:
                    out = model.generate(spatial_features=spatial, motion_features=motion)
                except Exception as e2:
                    warnings.warn(f"model.generate(...) with named args failed: {e2}")

        # 3) try calling forward() / __call__
        if out is None:
            tried.append("model(spatial, motion)")
            try:
                out = model(spatial, motion)
            except Exception as e:
                warnings.warn(f"Direct forward call model(spatial,motion) failed: {e}")
                # try other argument names
                try:
                    out = model(spatial_features=spatial, motion_features=motion)
                except Exception as e2:
                    warnings.warn(f"model(spatial_features=..., motion_features=...) failed: {e2}")

        # 4) If still None, try calling model.model (HF T5 model) if present
        if out is None and hasattr(model, "model"):
            tried.append("model.model.generate(...)")
            try:
                # need an input_ids / prompt — try building a prompt using the model.prompt or default
                prompt = getattr(model, "prompt", None) or getattr(model, "hparams", {}).get("prompt", "Translate the sentence.")
                # If tokenizer exists, encode prompt and call generate (this is a long-shot)
                if tokenizer is not None:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                    hf_model = getattr(model, "model")
                    out = hf_model.generate(input_ids=input_ids, max_length=args.max_tokens)
                else:
                    warnings.warn("No tokenizer for model.model.generate attempt")
            except Exception as e:
                warnings.warn(f"model.model.generate attempt failed: {e}")

    # Summarize what was attempted
    print("Tried inference entry points:", tried)

    # Interpret output
    if out is None:
        raise RuntimeError("All inference attempts failed. Inspect warnings above and the model API.")
    pred = extract_prediction_from_output(out, tokenizer, print_verbose=args.print_verbose)
    print("\n=== Prediction ===")
    print(pred)
    print("==================\n")

if __name__ == "__main__":
    main()
