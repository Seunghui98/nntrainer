# YOLOv7ReIDtiny — Pose + ReID on nntrainer

Port of the `YOLOv7ReIDtiny` model (RTMPose-style SimCC pose head + ReID
embedding head on a YOLOv7-tiny CSP backbone with a dual FPN neck) onto
nntrainer's functional graph API.

- **Backbone**: YOLOv7-tiny CSP, `widen_factor = 1.5` (stem 48 … deepest 768).
- **Neck**: two independent FPNs — `backbone.features` (pose) and
  `backbone.features_feat` (ReID) — each SPPCSPCTiny + 2× upsample + 2×
  downsample, single stride-32 end (768 ch @ 10×10).
- **Pose head** (`head`): 7×7 conv → flatten → ScaleNorm+Linear MLP → GAU
  (`rtmcc_gau` custom layer) → `cls_x` / `cls_y` SimCC classifiers → output
  `[1, 2·87, 640]` (`cls_x` rows stacked over `cls_y` rows).
- **ReID head** (`head_feat`): global-avg-pool → Linear → `[1, 128]`.

Input `1×3×320×320` (NCHW FP32). Pose decode is RTMPose SimCC: per keypoint,
`argmax` over 640 bins for x and y, `score = min(max_x, max_y)`, coordinates
divided by 2 (simcc split ratio).

## Files

| Path | Purpose |
|------|---------|
| `jni/yolov7_pose_graph.h` | Inline graph builders (backbone / neck / heads) |
| `jni/main.cpp` | `yolov7_pose_infer` — build, load, run, decode |
| `yolov7_pose.h` | `quick_ai::Model` wrapper for `nntr_quantize` |
| `../../layers/rtmcc_gau.{h,cpp}` | GAU (RTMCCBlock) custom layer |
| `../../res/yolov7_pose/` | reference model, weight converter, verify scripts |

## Build stages

Three precision stages, verified in order:

| Stage | Preset (`YOLO_TENSOR_TYPE`) | Weights | Activations | Layout |
|-------|-----|---------|-------------|--------|
| 1 | `w32a32` | FP32 | FP32 | NCHW |
| 2 | `w8a32` | Q8_0 | FP32 | NHWC |
| 3 (future) | `w8a16` | Q8_0 | FP16 | NHWC |

Stages 1 and 2 are implemented here. Channel-last (NHWC) is used from stage 2
on so every layer runs channel-last (no per-conv transposes).

## End-to-end (x86)

```bash
# 0. build (transformer feature gates the whole quick_ai tree)
meson setup build -Denable-transformer=true -Denable-app=true
ninja -C build Applications/quick_ai/models/YOLOv7Pose/jni/yolov7_pose_infer \
               Applications/quick_ai/nntr_quantize

cd Applications/quick_ai/res/yolov7_pose

# 1. FP32 weights: PyTorch .pt -> nntrainer safetensors
#    Accepts either a state_dict or a full torch.save(model) object; if the
#    checkpoint pickles the original training repo's classes (e.g. a missing
#    `models` package), the converter shims them to recover the state_dict.
#    Use --inspect first to print the checkpoint's raw keys/shapes and confirm
#    they match the reconstructed model.
python3 weight_converter.py --weights pose_base_v311.pt --inspect | head
python3 weight_converter.py --weights pose_base_v311.pt \
        --output /path/res/yolov7_pose.safetensors

# 1b. an input tensor (NCHW FP32, 1x3x320x320). Either export from the
#     reference inference.py, or (for a smoke test) generate one:
python3 make_reference.py --weights pose_merged_v311.pt --out /path/res

# 1c. stage 1 (W32A32) inference
YOLO_TENSOR_TYPE=w32a32 YOLO_WEIGHTS=/path/res/yolov7_pose.safetensors \
  ../../../../build/Applications/quick_ai/models/YOLOv7Pose/jni/yolov7_pose_infer \
  /path/res input_320.bin

# 2. Q8_0 weights via nntr_quantize (uses nntrainer's own packing)
../../../../build/Applications/quick_ai/nntr_quantize /path/res \
  --conv_dtype Q8_0 --output_format safetensors \
  --output_bin yolov7_pose_q8_0.safetensors -o /path/res

# 2b. stage 2 (W8A32) inference  -- ARM/Android only, see note below
YOLO_TENSOR_TYPE=w8a32 YOLO_WEIGHTS=/path/res/yolov7_pose_q8_0.safetensors \
  ./yolov7_pose_infer /path/res input_320.bin
```

> **Q8_0 runtime is ARM-only.** The quantized conv uses the NEON indirect-conv
> kernel (dotprod / i8mm); x86 has no NHWC quantized-conv fallback, so
> `w8a32` inference runs on the Android target, not on the x86 host. On x86 you
> can still (a) produce and inspect the Q8_0 safetensors with `nntr_quantize`
> and (b) verify stage-1 (`w32a32`) numerics. Stage-2 runtime parity is checked
> on device.

### Parity check

`make_reference.py` dumps the PyTorch reference (`ref_pose.bin`, `ref_reid.bin`)
and the input. Dump the nntrainer raw outputs and compare:

```bash
POSE_DUMP=/path/res/nn YOLO_TENSOR_TYPE=w32a32 \
  YOLO_WEIGHTS=/path/res/yolov7_pose.safetensors \
  yolov7_pose_infer /path/res input_320.bin
python3 verify_parity.py \
  --ref-pose /path/res/ref_pose.bin --ref-reid /path/res/ref_reid.bin \
  --out-pose /path/res/nn_pose.bin  --out-reid /path/res/nn_reid.bin
```

## Android (arm64-v8a)

```bash
export ANDROID_NDK=/path/to/android-ndk
cd Applications/quick_ai/models/YOLOv7Pose
./build_android.sh                 # libnntrainer + yolov7_pose_infer
./install_android.sh /path/res     # push binary + libs + weights + input
# then run on device (see script output)
```

## Note on the reconstructed reference

The upstream `models_pose/backbone/csp.py`, `head/keypoint.py`,
`head/reid.py` were not available; `res/yolov7_pose/models_pose/` contains a
reconstruction consistent with the provided modules and the published output
shapes. The weight converter maps by structural name, so if the real
checkpoint's module hierarchy differs, adjust the small mapping in
`weight_converter.py` (and the matching names in `yolov7_pose_graph.h`); the
`--dump-names` flag and `verify_parity.py` make mismatches obvious.
