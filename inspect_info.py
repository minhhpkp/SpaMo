# run_inspect_sample.py (run from repo root)
import numpy as np, pprint, os
path = "preprocess/Phoenix14T_real/train_info_ml.npy"
d = np.load(path, allow_pickle=True).item()

print("len =", len(d))
keys = list(d.keys())
print("first 10 keys (unsorted):", keys[:10])

# print prefix if exists
if 'prefix' in d:
    print("\n'prefix' key present. Value:")
    pprint.pprint(d['prefix'])

# find first numeric key and show its entry
num_keys = [k for k in keys if isinstance(k, int)]
num_keys.sort()
if num_keys:
    k0 = num_keys[0]
    print(f"\nFirst numeric key: {k0}")
    print("Entry keys and sample values:")
    entry = d[k0]
    if isinstance(entry, dict):
        for kk in entry.keys():
            val = entry[kk]
            # only print small values fully
            if kk in ('folder','fileid'):
                print(f"  {kk}: {val!r}")
            else:
                print(f"  {kk}: ({type(val).__name__})")
        print("\nFull entry (pretty):")
        pprint.pprint(entry)
else:
    print("No numeric keys found.")
