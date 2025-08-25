# inspect_model_api.py
import torch, inspect, os
from omegaconf import OmegaConf
from spamo.t5_slt import FlanT5SLT

CKPT = "./spamo.ckpt"    # <--- set to your checkpoint path
CFG  = "configs/finetune.yaml"      # <--- optional: used to pass model kwargs if needed

# load model with the same kwargs you used earlier
model_kwargs = {}
if os.path.exists(CFG):
    cfg = OmegaConf.load(CFG)
    if "model" in cfg and "params" in cfg.model:
        model_kwargs = OmegaConf.to_container(cfg.model.params, resolve=True)

print("Loading checkpoint... (may take a while)")
model = FlanT5SLT.load_from_checkpoint(CKPT, **model_kwargs, map_location="cpu")
model.eval()

print("\n--- Public methods and signatures ---")
for name, fn in inspect.getmembers(model, predicate=inspect.ismethod):
    if name.startswith("_"):
        continue
    try:
        print(name, inspect.signature(fn))
    except Exception:
        print(name, "(signature unavailable)")

print("\n--- Attributes of interest ---")
for attr in ("tokenizer","model","hparams","prompt","monitor","vocab"):
    if hasattr(model, attr):
        print(attr, "=>", type(getattr(model, attr)))
    else:
        print(attr, "=> MISSING")

# print class doc briefly
print("\nClass docstring (first 200 chars):")
print((model.__class__.__doc__ or "")[:200])
