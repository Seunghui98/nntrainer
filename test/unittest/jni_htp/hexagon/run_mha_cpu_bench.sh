#!/usr/bin/env bash
#
# Build and run the ARM CPU baseline for the MHA-core GEMMs (mha_cpu_bench.cpp)
# on-device, alongside hexkl_micro_mha_bench.c's DSP numbers. Does NOT modify
# Applications/CausalLM/layers/mha_core.cpp -- cross-compiles a small
# standalone binary with the NDK and links it directly against an
# already-built libnntrainer.so (no meson/ndk-build of the whole library
# needed), matching the exact fp16 compute_kcaches / compute_fp16vcache_transposed
# symbols that file calls.
#
# Usage:   ./run_mha_cpu_bench.sh
# Override via env, e.g. ANDROID_NDK=... NNTR_LIBDIR=... DEVICE=XXXX ./run_mha_cpu_bench.sh
#
set -euo pipefail

ANDROID_NDK=${ANDROID_NDK:-$HOME/workspace/android-ndk-r26d}
NNTR=${NNTR:-$HOME/workspace/nntrainer}
# Prebuilt arm64-v8a libnntrainer.so + headers this links against. Must be
# built with -DENABLE_FP16=1 -DUSE__FP16=1 (checked below) -- that gates
# whether _FP16 in the public headers is __fp16 or _Float16, and this file
# hardcodes __fp16 to match.
NNTR_LIBDIR=${NNTR_LIBDIR:-$NNTR/builddir/android_build_result/lib/arm64-v8a}
CPU_OS=${CPU_OS:-29}
DEVICE=${DEVICE:-$(adb devices | awk 'NR==2{print $1}')}
DEV_DIR=${DEV_DIR:-/data/local/tmp/mha_cpu_bench}

say() { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }

[ -n "$DEVICE" ] || { echo "no adb device found (set DEVICE=serial)"; exit 1; }
SRC="$NNTR/test/unittest/jni_htp/hexagon/mha_cpu_bench.cpp"
[ -f "$SRC" ] || { echo "source missing: $SRC"; exit 1; }
[ -f "$NNTR_LIBDIR/libnntrainer.so" ] || {
  echo "libnntrainer.so not found at $NNTR_LIBDIR -- set NNTR_LIBDIR to an"
  echo "arm64-v8a build (e.g. from tools/package_android.sh's output, or"
  echo "builddir/android_build_result if present)."
  exit 1
}

CXX="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${CPU_OS}-clang++"
[ -x "$CXX" ] || { echo "NDK clang++ not found at $CXX (check ANDROID_NDK/CPU_OS)"; exit 1; }

say "device=$DEVICE  ndk=$ANDROID_NDK  libnntrainer=$NNTR_LIBDIR"

# build output goes OUTSIDE the repo tree by default (override via BUILD_DIR)
# so a plain run doesn't leave untracked build artifacts under test/.
BUILD_DIR=${BUILD_DIR:-/tmp/mha_cpu_bench_build}
mkdir -p "$BUILD_DIR"
BIN="$BUILD_DIR/mha_cpu_bench"

say "building (NDK, links prebuilt libnntrainer.so, no meson/ndk-build of the whole library)"
"$CXX" -std=c++17 -O3 -march=armv8.2-a+fp16+dotprod+i8mm -DENABLE_FP16=1 -DUSE__FP16=1 \
  "$SRC" -o "$BIN" -L"$NNTR_LIBDIR" -lnntrainer -Wl,-rpath,'$ORIGIN'
echo "    $BIN"

say "pushing binary + shared libs"
adb -s "$DEVICE" shell "mkdir -p $DEV_DIR"
adb -s "$DEVICE" push "$BIN" "$DEV_DIR/mha_cpu_bench" >/dev/null
for lib in libnntrainer.so libc++_shared.so libsdkl.so; do
  [ -f "$NNTR_LIBDIR/$lib" ] && adb -s "$DEVICE" push "$NNTR_LIBDIR/$lib" "$DEV_DIR/$lib" >/dev/null
done
adb -s "$DEVICE" shell "chmod 755 $DEV_DIR/mha_cpu_bench"

say "running on device (this takes ~1-2 minutes: prefill chunks run 20x20 total row-computations)"
RAW=$(adb -s "$DEVICE" shell "cd $DEV_DIR && LD_LIBRARY_PATH=$DEV_DIR ./mha_cpu_bench")
echo "$RAW"

say "RESULTS (ARM CPU baseline, matches hexkl_micro_mha_bench.c's shapes)"
echo "$RAW" | awk '
  /^KCACHE_DEC_1HEAD_US_kv1024/     { print "  qkT_dec_1024  1 kv_head, single thread:      " $2 " us" }
  /^KCACHE_DEC_8HEAD_SEQ_US_kv1024/ { print "  qkT_dec_1024  8 kv_heads, single thread:      " $2 " us  <- decode Q.K^T CPU bar" }
  /^VCACHE_DEC_8HEAD_SEQ_US_kv1024/ { print "  pv_dec_1024   8 kv_heads, single thread:      " $2 " us  <- decode P.V CPU bar (kv=1024)" }
  /^VCACHE_DEC_8HEAD_SEQ_US_kv512/  { print "  pv_dec_512    8 kv_heads, single thread:      " $2 " us  <- decode P.V CPU bar (kv=512)" }
  /^PREFILL_QK_CPU_SEQ_US/          { print "  qk_pre_128    single-thread:                  " $2 " us/chunk" }
  /^PREFILL_QK_CPU_PAR_US/          { print "  qk_pre_128    8-thread (pooled):               " $2 " us/chunk  <- prefill Q.K^T CPU bar (scaling not fully explained, see source comment)" }
  /^PREFILL_PV_CPU_SEQ_US/          { print "  pv_pre_128    single-thread:                  " $2 " us/chunk" }
  /^PREFILL_PV_CPU_PAR_US/          { print "  pv_pre_128    8-thread (pooled):               " $2 " us/chunk  <- prefill P.V CPU bar" }
'

cat <<'NOTE'

--------------------------------------------------------------------
These are the "CPU" bars in hexkl_micro_mha_bench.c's consolidated report.
Run-to-run variance is real on this device (DVFS/scheduling) -- run this a
few times and look at the range, not a single sample, before trusting a
specific number in a doc.

Prefill's 8-thread parallel scaling (~1.6-3.0x, not the ~8x a compute-bound
kernel "should" show) is NOT a settled result -- it survived the M2-n fix
(hoisting thread spawn out of the timed loop), so it looks like a real
ceiling and not a spawn-overhead artifact, but the mechanism (bandwidth
contention vs thermal throttling under sustained multi-core load) is not
pinned down. Treat it as reported, not explained.
--------------------------------------------------------------------
NOTE
