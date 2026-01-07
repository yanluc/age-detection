#!/usr/bin/env python3
import argparse
import os
import re
import numpy as np
import torch

from model import AgeRegressionCNN

def _to_c_float_list(x: np.ndarray) -> str:
    # Compact but readable; you can tweak formatting if you want.
    flat = x.reshape(-1)
    return ", ".join(f"{v:.9g}f" for v in flat)

def _write_array(f, name: str, arr: np.ndarray):
    f.write(f"// {name}: shape={tuple(arr.shape)}\n")
    f.write(f"static const float {name}[] = {{\n")
    s = _to_c_float_list(arr.astype(np.float32))
    # Wrap lines to keep compiler happier
    line_len = 0
    f.write("  ")
    for tok in s.split(", "):
        chunk = tok + ", "
        if line_len + len(chunk) > 100:
            f.write("\n  ")
            line_len = 0
        f.write(chunk)
        line_len += len(chunk)
    f.write("\n};\n\n")

def load_state_dict_any(path: str):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["model_state"]
    if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
        # likely a raw state_dict
        return obj
    raise RuntimeError("Unsupported checkpoint format. Provide checkpoint.pt (with model_state) or a raw state_dict .pth")

def infer_base_channels(sd: dict) -> int:
    # features.0.0.weight shape: [out_ch, in_ch, 3, 3] where out_ch=base_channels
    k = "features.0.0.weight"
    if k not in sd:
        raise RuntimeError(f"Can't infer base_channels; missing key {k}")
    return int(sd[k].shape[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint.pt or state_dict.pth")
    ap.add_argument("--out", default="weights.h", help="Output header path")
    args = ap.parse_args()

    sd = load_state_dict_any(args.ckpt)
    base = infer_base_channels(sd)

    model = AgeRegressionCNN(base_channels=base)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # Map layer names for your exact architecture in model.py
    conv_names = [
        ("features.0.0", "CONV0"),  # 3 -> base
        ("features.0.2", "CONV1"),  # base -> base
        ("features.1.0", "CONV2"),  # base -> 2base
        ("features.1.2", "CONV3"),  # 2base -> 2base
        ("features.2.0", "CONV4"),  # 2base -> 4base
        ("features.2.2", "CONV5"),  # 4base -> 4base
        ("features.3.0", "CONV6"),  # 4base -> 8base
        ("features.3.2", "CONV7"),  # 8base -> 8base
    ]
    fc_names = [
        ("regressor.1", "FC0"),     # (8base) -> 128
        ("regressor.4", "FC1"),     # 128 -> 64
        ("regressor.7", "FC2"),     # 64 -> 1
    ]

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("#pragma once\n")
        f.write("// Auto-generated from PyTorch weights by export_weights.py\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"static const int kBaseChannels = {base};\n")
        f.write("static const int kInputH = 200;\n")
        f.write("static const int kInputW = 200;\n")
        f.write("static const int kInputC = 3;\n\n")

        # Convs
        for prefix, cname in conv_names:
            w = model.state_dict()[f"{prefix}.weight"].cpu().numpy()  # [out,in,3,3]
            b = model.state_dict()[f"{prefix}.bias"].cpu().numpy()    # [out]
            _write_array(f, f"{cname}_W", w)
            _write_array(f, f"{cname}_B", b)

        # FCs
        for prefix, fname in fc_names:
            w = model.state_dict()[f"{prefix}.weight"].cpu().numpy()  # [out,in]
            b = model.state_dict()[f"{prefix}.bias"].cpu().numpy()    # [out]
            _write_array(f, f"{fname}_W", w)
            _write_array(f, f"{fname}_B", b)

    print(f"[OK] Wrote {args.out}")
    print(f"Base channels = {base}")
    print("Next: copy weights.h into your Arduino sketch folder and compile.")

if __name__ == "__main__":
    main()
