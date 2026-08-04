#!/usr/bin/env bash
#
# One-shot: build the micro-API fp16 MHA-core benchmark (Q.K^T / P.V shapes),
# run it on the Hexagon DSP, and print the shape/kernel report.
#
# Usage:   ./run_micro_mha_bench.sh
# Override any of these via env, e.g.  HEX_ARCH=v75 DEVICE=XXXX ./run_micro_mha_bench.sh
#
set -euo pipefail

# ---------------- config (override via env) ----------------
HEXAGON_SDK=${HEXAGON_SDK:-$HOME/workspace/Hexagon_SDK/6.4.0.2}
ADDON=${ADDON:-$HOME/workspace/hxkl-beta2/hexkl_addon}
NNTR=${NNTR:-$HOME/workspace/nntrainer}
HEX_ARCH=${HEX_ARCH:-v79}
ARM_ARCH=${ARM_ARCH:-armv8}
CPU_OS=${CPU_OS:-android26}
DEVICE=${DEVICE:-$(adb devices | awk 'NR==2{print $1}')}

SRC="$NNTR/test/unittest/jni_htp/hexagon/hexkl_micro_mha_bench.c"
BASE="$ADDON/examples/hexkl_micro_hmx_mm_f16"   # fp16 example: link line + flags already suit fp16
EXDIR="$ADDON/examples/hexkl_mha_probe"         # dir name must start with hexkl_ (run_main path)

say() { printf '\n\033[1;36m>>> %s\033[0m\n' "$*"; }

[ -n "$DEVICE" ] || { echo "no adb device found (set DEVICE=serial)"; exit 1; }
[ -f "$SRC" ]    || { echo "benchmark source missing: $SRC"; exit 1; }
say "device=$DEVICE  hex=$HEX_ARCH  arm=$ARM_ARCH/$CPU_OS"

# ---------------- 1. Hexagon SDK env (zsh/leftover-guard safe) ----------------
say "sourcing Hexagon SDK env"
# the vendor env script is not written for 'set -eu'; relax around the source
unset HEXAGON_SDK_ROOT || true
set +eu
cd "$HEXAGON_SDK"
source ./setup_sdk_env.source >/dev/null 2>&1
set -eu
[ -n "${DEFAULT_HEXAGON_TOOLS_ROOT:-}" ] || { echo "SDK env failed - run 'unset HEXAGON_SDK_ROOT' then source setup_sdk_env.source manually"; exit 1; }
echo "    tools: $DEFAULT_HEXAGON_TOOLS_ROOT ($DEFAULT_TOOLS_VARIANT)"

# ---------------- 2. stage the example dir (idempotent) ----------------
say "staging $EXDIR"
[ -d "$EXDIR" ] || cp -r "$BASE" "$EXDIR"
rm -f "$EXDIR"/src/*.c
cp "$SRC" "$EXDIR/src/test_hexkl_mha_probe.c"   # name must match dir: test_<dir>.c
# turn on the on-target HAP_perf timer branch (once)
grep -q USE_HAP_PERF "$EXDIR/build.sh" || \
  sed -i 's|NPU_OPT_FLAGS="-fpic -mhvx|NPU_OPT_FLAGS="-DUSE_HAP_PERF -fpic -mhvx|' "$EXDIR/build.sh"
# also link libhexkl_macro.a (SDKL kernel) alongside libhexkl_micro.a (once)
grep -q "libhexkl_macro.a" "$EXDIR/build.sh" || \
  sed -i 's|"$MICRO_LINK_PATHS/libhexkl_micro.a"|"$MICRO_LINK_PATHS/libhexkl_micro.a" "$MICRO_LINK_PATHS/libhexkl_macro.a"|' "$EXDIR/build.sh"
# opt-in: MHA_RUN_DIAGNOSTICS=1 ./run_micro_mha_bench.sh includes the DMA
# bandwidth-ceiling sweep and the output narrow-vs-full comparison (M2-f/
# M2-j) -- diagnostic, NOT part of any shape's headline number, off by
# default so a plain run's output matches the consolidated report exactly.
if [ "${MHA_RUN_DIAGNOSTICS:-0}" = "1" ]; then
  grep -q "MHA_RUN_DIAGNOSTICS" "$EXDIR/build.sh" || \
    sed -i 's|NPU_OPT_FLAGS="-DUSE_HAP_PERF -fpic -mhvx|NPU_OPT_FLAGS="-DUSE_HAP_PERF -DMHA_RUN_DIAGNOSTICS=1 -fpic -mhvx|' "$EXDIR/build.sh"
  say "MHA_RUN_DIAGNOSTICS=1: diagnostic sweeps WILL run (not part of the headline numbers)"
else
  # if a previous run left the flag on, take it back out so the default is
  # really the default.
  sed -i 's|-DMHA_RUN_DIAGNOSTICS=1 ||' "$EXDIR/build.sh" 2>/dev/null || true
fi

# ---------------- 3. build (hexagon-clang -> DSP .so) ----------------
say "building for $HEX_ARCH"
cd "$EXDIR"
./build.sh --hex-arch "$HEX_ARCH"
SO="build/${HEXKL_SDK_VER:-6.4.0.2}/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/libtest_hexkl_mha_probe_q.so"
[ -f "$SO" ] || { echo "build produced no .so"; exit 1; }
echo "    $SO"

# ---------------- 4. run on device (run_main_on_hexagon) ----------------
say "running on DSP"
export ADB_FLAGS="-s $DEVICE"
./run_android.sh --hex-arch "$HEX_ARCH" --arm-arch "$ARM_ARCH" --cpu-os "$CPU_OS" 2>&1 \
  | grep -iE "Successfully|return value|Error|FAIL" || true

# ---------------- 5. report ----------------
LOG=$(ls -t logs/*/hexagon_*_"${HEX_ARCH}"/run_android_log.txt 2>/dev/null | head -1)
[ -f "$LOG" ] || { echo "no log captured"; exit 1; }

say "RESULTS  (fp16 MHA-core micro/HMX bench; on-device us via HAP_perf)"
# Consolidated report. Keyed by shape NAME via the self-documenting
# "MHA_FIELD shape=<name> field=<name> value=<v>" lines each shape driver
# prints once at the end (field names embed their own units, e.g.
# "_per_head" vs "_8heads" -- this is exactly what a bare, unlabeled
# microseconds column invited the wrong reading on before). M/N/K still come
# from each shape's own "MHA_SHAPE name=... M=... N=... K=..." banner.
#
# Units, spelled out because a column of microseconds with no unit stated
# is exactly what caused the last confusion:
#   *_per_head       : cost of ONE kv_head's worth of work (single call)
#   *_8heads         : cost of ALL 8 kv_heads (a full layer), the number
#                       that's actually comparable to the CPU baseline
#   TOTAL (us/layer)  : the headline -- optimized path + incremental bake
#                       where the shape needs one (prefill always does;
#                       decode's P.V does; decode's Q.K^T does not, since a
#                       long-running decode session can assume an
#                       already-baked cache from prior turns, see the
#                       source comments for why that assumption is NOT
#                       valid at prefill)
sed 's/.*CDSP0:\[SU\]: //' "$LOG" | awk '
  /^MHA_SHAPE / {
    nm=$0
    if (match(nm,/name=[^ ]+/)) cur=substr(nm,RSTART+5,RLENGTH-5)
    if (!(cur in seen)) { seen[cur]=1; ni++; order[ni]=cur }
    if (match(nm,/M=[0-9]+/)) MM[cur]=substr(nm,RSTART+2,RLENGTH-2)
    if (match(nm,/N=[0-9]+/)) NN[cur]=substr(nm,RSTART+2,RLENGTH-2)
    if (match(nm,/K=[0-9]+/)) KK[cur]=substr(nm,RSTART+2,RLENGTH-2)
    next
  }
  /^MHA_FIELD / {
    nm=$0; sh=""; fl=""; vl=""
    if (match(nm,/shape=[^ ]+/)) sh=substr(nm,RSTART+6,RLENGTH-6)
    if (match(nm,/field=[^ ]+/)) fl=substr(nm,RSTART+6,RLENGTH-6)
    if (match(nm,/value=[^ ]+/)) vl=substr(nm,RSTART+6,RLENGTH-6)
    val[sh,fl]=vl
    next
  }
  END{
    print ""
    print "| shape        |    M |    N |    K | macro (us/head) | micro naive (us/head) | micro+DMA/tile (us/head) | micro+layerDMA (us/layer,8h) | +incr bake (us/layer,8h) | TOTAL (us/layer,8h) | GB/s | max relErr | correct |"
    print "|--------------|-----:|-----:|-----:|----------------:|-----------------------:|--------------------------:|------------------------------:|---------------------------:|---------------------:|-----:|-----------:|:-------:|"
    for (i=1;i<=ni;i++) {
      k=order[i]
      mac  = ((k SUBSEP "macro_us_per_head") in val)          ? val[k,"macro_us_per_head"]          : "-"
      mic  = ((k SUBSEP "micro_us_per_head") in val)          ? val[k,"micro_us_per_head"]          : "-"
      dpt  = ((k SUBSEP "dma_pertile_us_per_head") in val)    ? val[k,"dma_pertile_us_per_head"]    : "-"
      dly  = ((k SUBSEP "dma_layer_us_8heads") in val)        ? val[k,"dma_layer_us_8heads"]        : "-"
      inc  = ((k SUBSEP "incr_bake_us_8heads") in val)        ? val[k,"incr_bake_us_8heads"]        : "-"
      tot  = ((k SUBSEP "total_us_8heads") in val)            ? val[k,"total_us_8heads"]            : "-"
      gp   = ((k SUBSEP "gbps") in val)                       ? val[k,"gbps"]                       : "-"
      re   = ((k SUBSEP "maxrelerr") in val)                  ? val[k,"maxrelerr"]                  : "-"
      co   = ((k SUBSEP "correct") in val)                    ? val[k,"correct"]                    : "-"
      printf "| %-12s | %4s | %4s | %4s | %15s | %22s | %26s | %30s | %27s | %21s | %4s | %10s | %7s |\n",
        k, MM[k], NN[k], KK[k], mac, mic, dpt, dly, inc, tot, gp, re, co
    }
  }'

cat <<NOTE

--------------------------------------------------------------------
All 5 shapes above run, and are measured, on every plain invocation with no
env overrides: decode Q.K^T (qkT_dec_1024), decode P.V (pv_dec_1024,
pv_dec_512), prefill Q.K^T (qk_pre_128), prefill P.V (pv_pre_128). The
"macro"/"micro naive"/"micro+DMA/tile" columns only exist for qkT_dec_1024
(they are the historical baselines that motivated every fix in this file --
kept as explicit comparison columns, not dropped, because their slowness is
itself a finding); other shapes show "-" there, not a fabricated number.

Full per-shape phase breakdown (weight DMA / WH bake, incremental vs full /
activation stage / mm_f16 / acc_read+ah_to_rm / output DMA) is in the raw
log above the table.

Diagnostic sweeps (DMA bandwidth ceiling, output narrow-vs-full) -- NOT part
of any shape's headline number -- $([ "${MHA_RUN_DIAGNOSTICS:-0}" = "1" ] && echo "ran this time (MHA_RUN_DIAGNOSTICS=1)." || echo "were SKIPPED (the default). Set MHA_RUN_DIAGNOSTICS=1 to include them.")

The ARM CPU baseline (the numbers these DSP totals are compared against) is
a SEPARATE checked-in harness: run_mha_cpu_bench.sh / mha_cpu_bench.cpp in
this same directory. Run it separately -- it is not part of this script,
since it targets the CPU, not the Hexagon DSP, and has its own build/push
path (NDK cross-compile against a prebuilt libnntrainer.so, no meson).
--------------------------------------------------------------------
NOTE
