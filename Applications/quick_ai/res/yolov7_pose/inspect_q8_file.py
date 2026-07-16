#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Inspect an nntrainer-produced (quantized) .safetensors file for corruption.

nntr_quantize writes:
  * plain tensors (head weights, biases, non-eligible convs) as "F32"/"F16"
  * Q8_0 conv weights as a "U8" byte blob with an "nntr_dtype":"Q8_0" extension,
    physically laid out as block_q8_0x4 super-blocks (4 fp16 scales + 128 int8,
    the 4-column interleave produced by repack_q8_0 at save time).

This checks every tensor for non-finite values *without* nntrainer or a rebuild:
  - F32/F16 tensors: report nan/inf/min/max over the raw values.
  - Q8_0 tensors: decode the fp16 block scales + int8 payload and report the
    dequantized min/max plus any non-finite scale, so a bad quantize/repack or a
    truncated blob is caught directly on the file.

Usage:
  python3 inspect_q8_file.py yolov7_pose_q8_0.safetensors
  python3 inspect_q8_file.py yolov7_pose_q8_0.safetensors --only head   # substr
"""
import argparse
import json
import struct
import sys

import numpy as np


def load_header(path):
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(hlen).decode("utf-8"))
        base = 8 + hlen
    return header, base


def read_span(path, base, off):
    with open(path, "rb") as f:
        f.seek(base + off[0])
        return f.read(off[1] - off[0])


def dequant_q8_0x4(raw, out_ch, crs):
    """Decode block_q8_0x4 [out_ch/4][crs/32] -> float32 [out_ch, crs]."""
    nb = crs // 32
    sb = out_ch // 4
    sb_bytes = 8 + 128  # 4x fp16 scale + 128 int8
    expect = sb * nb * sb_bytes
    if len(raw) != expect:
        return None, f"byte-size mismatch: have {len(raw)} expect {expect}"
    buf = np.frombuffer(raw, dtype=np.uint8)
    out = np.zeros((out_ch, crs), dtype=np.float32)
    bad_scale = 0
    pos = 0
    for s in range(sb):
        for j in range(nb):
            d = np.frombuffer(raw[pos : pos + 8], dtype=np.float16).astype(np.float32)
            qs = np.frombuffer(raw[pos + 8 : pos + 8 + 128], dtype=np.int8)
            pos += sb_bytes
            if not np.isfinite(d).all():
                bad_scale += int((~np.isfinite(d)).sum())
            for r in range(4):
                row = s * 4 + r
                for sub in range(4):
                    seg = qs[32 * sub + 8 * r : 32 * sub + 8 * r + 8].astype(np.float32)
                    out[row, j * 32 + sub * 8 : j * 32 + sub * 8 + 8] = seg * d[r]
    return out, (f"non-finite-scales={bad_scale}" if bad_scale else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--only", default="", help="only tensors whose name contains this")
    args = ap.parse_args()

    header, base = load_header(args.file)
    names = [k for k in header if k != "__metadata__"]
    names.sort()

    n_bad = 0
    for name in names:
        if args.only and args.only not in name:
            continue
        e = header[name]
        dt = e.get("nntr_dtype", e["dtype"])
        shape = e.get("shape", [])
        off = e["data_offsets"]
        raw = read_span(args.file, base, off)
        note = ""
        if dt in ("F32", "FP32"):
            v = np.frombuffer(raw, dtype=np.float32)
        elif dt in ("F16", "FP16"):
            v = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
        elif dt == "Q8_0":
            # shape is [1,1,CRS,out_ch] (matmul weight); decode to [out_ch,CRS].
            crs, out_ch = int(shape[-2]), int(shape[-1])
            v, note = dequant_q8_0x4(raw, out_ch, crs)
            if v is None:
                print(f"  !! {name:50s} {dt:6s} {str(shape):20s} {note}")
                n_bad += 1
                continue
        else:
            print(f"  -- {name:50s} {dt:6s} {str(shape):20s} (skipped)")
            continue

        nnan = int(np.isnan(v).sum())
        ninf = int(np.isinf(v).sum())
        fin = v[np.isfinite(v)]
        mn = float(fin.min()) if fin.size else float("nan")
        mx = float(fin.max()) if fin.size else float("nan")
        flag = "!!" if (nnan or ninf or note) else "ok"
        if nnan or ninf or note:
            n_bad += 1
        print(
            f"  {flag} {name:50s} {dt:6s} {str(shape):20s} "
            f"nan={nnan} inf={ninf} min={mn:+.4g} max={mx:+.4g} {note}"
        )

    print(f"\n{'CORRUPT' if n_bad else 'CLEAN'}: {n_bad} tensor(s) with non-finite/bad data")
    sys.exit(1 if n_bad else 0)


if __name__ == "__main__":
    main()
