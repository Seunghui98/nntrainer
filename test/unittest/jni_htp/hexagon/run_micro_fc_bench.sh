#!/usr/bin/env bash
#
# One-shot: build the micro-API u8i4 + u8i8 fc benchmark, run it on the
# Hexagon DSP, and print the fc_compare-format breakdown -- DSP matmul vs
# scalar/DMA staging, cross-matmul weight-prefetch vs the SDKL macro kernel,
# u8i4 next to u8i8 in one table (same DSP session, same thermal state).
#
# Usage:   ./run_micro_fc_bench.sh
# Override any of these via env, e.g.  HEX_ARCH=v75 DEVICE=XXXX ./run_micro_fc_bench.sh
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

SRC="$NNTR/test/unittest/jni_htp/hexagon/hexkl_micro_fc_bench.c"
BASE="$ADDON/examples/hexkl_micro_hmx_mm_u8i4_i32"   # provides build/run scripts + micro lib link
EXDIR="$ADDON/examples/hexkl_vtcm_probe"             # dir name must start with hexkl_ (run_main path)

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
cp "$SRC" "$EXDIR/src/test_hexkl_vtcm_probe.c"   # name must match dir: test_<dir>.c
# turn on the on-target HAP_perf timer branch (once)
grep -q USE_HAP_PERF "$EXDIR/build.sh" || \
  sed -i 's|NPU_OPT_FLAGS="-fpic -mhvx|NPU_OPT_FLAGS="-DUSE_HAP_PERF -fpic -mhvx|' "$EXDIR/build.sh"
# also link libhexkl_macro.a (SDKL kernel, both u8i4 and u8i8 symbols live in
# the same archive) alongside libhexkl_micro.a (once)
grep -q "libhexkl_macro.a" "$EXDIR/build.sh" || \
  sed -i 's|"$MICRO_LINK_PATHS/libhexkl_micro.a"|"$MICRO_LINK_PATHS/libhexkl_micro.a" "$MICRO_LINK_PATHS/libhexkl_macro.a"|' "$EXDIR/build.sh"

# ---------------- 3. build (hexagon-clang -> DSP .so) ----------------
say "building for $HEX_ARCH"
cd "$EXDIR"
./build.sh --hex-arch "$HEX_ARCH"
SO="build/${HEXKL_SDK_VER:-6.4.0.2}/hexagon_${DEFAULT_TOOLS_VARIANT}_${HEX_ARCH}/libtest_hexkl_vtcm_probe_q.so"
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

say "RESULTS  (fc_compare format; on-device us via HAP_perf; u8i4 vs u8i8, same session)"
#
# Every number below comes from a self-describing marker line the bench
# prints once per (dtype, shape, field):
#
#   FC_FIELD dtype=u8i8 shape=q_proj field=cross_us value=61.2
#
# field names embed their own units/meaning, so the awk below looks fields
# up by name instead of by column position -- the MHA bench's report script
# once silently dropped every shape past its first hardcoded milestone name
# and printed a stale column instead; keying strictly by field name is what
# stops that class of bug from recurring here. A plain invocation with no
# env overrides runs and prints EVERY (dtype, shape) pair below, including
# the two optional/secondary rows from spec §5 -- nothing is gated off by
# default; there is nothing slow enough in this bench to need an opt-in flag.
sed 's/.*CDSP0:\[SU\]: //' "$LOG" | awk \
  -v REL="${REL_ERR:-0.07207}" \
  -v HOST="${HOST_NEON_US:-76.7}" \
  -v SDKL_PROD_NPU="${SDKL_PROD_NPU_US:-270.0}" \
  -v SDKL_PROD_TOT="${SDKL_PROD_TOT_US:-350.5}" \
  -v QNN="${QNN_FC_US:-69.9}" '
  /^FC_FIELD / {
    nm=$0; dt=""; sh=""; fl=""; vl=""
    if (match(nm,/dtype=[^ ]+/)) dt=substr(nm,RSTART+6,RLENGTH-6)
    if (match(nm,/shape=[^ ]+/)) sh=substr(nm,RSTART+6,RLENGTH-6)
    if (match(nm,/field=[^ ]+/)) fl=substr(nm,RSTART+6,RLENGTH-6)
    if (match(nm,/value=[^ ]+/)) vl=substr(nm,RSTART+6,RLENGTH-6)
    val[dt,sh,fl]=vl
    if (!(dt in dtseen))   { dtseen[dt]=1;   ndt++;   dtorder[ndt]=dt }
    if (!(sh in shseen))   { shseen[sh]=1;   nsh++;   shorder[nsh]=sh }
    next
  }
  /^\[.*\] M=/ {
    nm=$0; sub(/].*/,"",nm); sub(/\[/,"",nm)
    if(match(nm,/M=[0-9]+/)) MM[nm]=substr(nm,RSTART+2,RLENGTH-2)  # unused fallback, kept harmless
    next
  }
  END{
    # canonical shape/dtype order -- SHAPES[] / DTYPE_OPS order in the .c file
    shord[1]="sample"; shord[2]="small"; shord[3]="medium"; shord[4]="q_proj"; shord[5]="ffn_up"; nshc=5
    dtord[1]="u8i4"; dtord[2]="u8i8"; ndtc=2

    print ""
    print "=== Core comparison: u8i4 vs u8i8, per shape (NPU kernel only, on-DSP; host quant/dequant excluded both sides) ==="
    print ""
    print "| shape  | dtype | SDKL macro us | single (micro+DMA) us | cross-matmul us | speedup vs SDKL | correct | weight-DMA GB/s |"
    print "|--------|-------|--------------:|-----------------------:|-----------------:|-----------------:|:-------:|-----------------:|"
    for (si=1; si<=nshc; si++) {
      sh=shord[si]
      for (di=1; di<=ndtc; di++) {
        dt=dtord[di]
        sdkl   = ((dt,sh,"sdkl_us") in val)          ? val[dt,sh,"sdkl_us"]          : "-"
        single = ((dt,sh,"single_us") in val)        ? val[dt,sh,"single_us"]        : "-"
        cross  = ((dt,sh,"cross_us") in val)         ? val[dt,sh,"cross_us"]         : "-"
        corr   = ((dt,sh,"cross_correct") in val)    ? val[dt,sh,"cross_correct"]    : "-"
        gbps   = ((dt,sh,"weight_dma_gbps") in val)  ? val[dt,sh,"weight_dma_gbps"]  : "-"
        sp = "-"
        if (sdkl != "-" && cross != "-" && cross+0 > 0) sp = sprintf("%.2fx", sdkl/cross)
        printf "| %-6s | %-5s | %13s | %23s | %17s | %16s | %-7s | %16s |\n",
          sh, dt, sdkl, single, cross, sp, corr, gbps
      }
    }

    print ""
    print "=== §3 prediction check (q_proj headline) ==="
    print ""
    c4=val["u8i4","q_proj","cross_us"]+0; c8=val["u8i8","q_proj","cross_us"]+0
    s4=val["u8i4","q_proj","single_us"]+0; s8=val["u8i8","q_proj","single_us"]+0
    printf "u8i4 cross=%.1f us | u8i8 cross=%.1f us | delta=%+.1f us (%+.1f%%)\n", c4, c8, c8-c4, (c4>0?100.0*(c8-c4)/c4:0)
    printf "u8i4 single=%.1f us | u8i8 single=%.1f us | delta=%+.1f us (%+.1f%%)\n", s4, s8, s8-s4, (s4>0?100.0*(s8-s4)/s4:0)
    print "doc 13 predicted: cross 58-65us (5-15% worse), single ~134us for u8i8 q_proj. Compare the deltas above against that band."

    print ""
    print "=== §5 optional/secondary rows (kept separate from the core table above; NOT part of the u8i4-vs-u8i8 verdict) ==="
    print ""
    print "-- 5.1 activation DMA row size (bandwidth-only; dtype-independent, dtype=act) --"
    print "| shape  | 32B-row us | 32B-row GB/s | one-shot us | one-shot GB/s |"
    print "|--------|-----------:|-------------:|------------:|---------------:|"
    for (si=1; si<=nshc; si++) {
      sh=shord[si]
      sm  = (("act",sh,"opt_act_smallrow_us") in val)   ? val["act",sh,"opt_act_smallrow_us"]   : "-"
      smg = (("act",sh,"opt_act_smallrow_gbps") in val) ? val["act",sh,"opt_act_smallrow_gbps"] : "-"
      bg  = (("act",sh,"opt_act_bigrow_us") in val)     ? val["act",sh,"opt_act_bigrow_us"]     : "-"
      bgg = (("act",sh,"opt_act_bigrow_gbps") in val)   ? val["act",sh,"opt_act_bigrow_gbps"]   : "-"
      if (sm != "-" || bg != "-")
        printf "| %-6s | %10s | %13s | %11s | %14s |\n", sh, sm, smg, bg, bgg
    }
    print ""
    print "-- 5.2 ring_init lite (per-matmul 32KB memset removed from the cross-matmul path) --"
    print "| shape  | dtype | cross (heavy) us | cross (ring_lite) us | delta us | correct |"
    print "|--------|-------|------------------:|-----------------------:|---------:|:-------:|"
    for (si=1; si<=nshc; si++) {
      sh=shord[si]
      for (di=1; di<=ndtc; di++) {
        dt=dtord[di]
        heavy = ((dt,sh,"cross_us") in val) ? val[dt,sh,"cross_us"] : "-"
        lite  = ((dt,sh,"opt_ring_lite_us") in val) ? val[dt,sh,"opt_ring_lite_us"] : "-"
        lok   = ((dt,sh,"opt_ring_lite_correct") in val) ? val[dt,sh,"opt_ring_lite_correct"] : "-"
        dl = "-"
        if (heavy != "-" && lite != "-") dl = sprintf("%.1f", lite-heavy)
        if (heavy != "-" || lite != "-")
          printf "| %-6s | %-5s | %17s | %23s | %8s | %-7s |\n", sh, dt, heavy, lite, dl, lok
      }
    }

    print ""
    vt = val["all","all","vtcm_size_bytes"]
    if (vt != "") printf "VTCM size reported by hexkl_micro_hw_init: %s bytes (%.2f MiB)\n", vt, vt/1048576.0

    qi_dt="u8i4"; qi_sh="q_proj"
    c=val[qi_dt,qi_sh,"cross_us"]+0; s=val[qi_dt,qi_sh,"sdkl_us"]+0
    if (c>0) {
      percall=c+HOST; sdkl_dsp_tot=s+HOST
      print ""
      print "fc_compare-format detail (u8i4 q_proj, matches hexkl_fc_compare layout, acceptance-gate shape):"
      print ""
      printf "[u8i4 cross-matmul (micro+DMA skel)]  proj=q_proj  relErr=%s  engine=NPU/HMX\n",REL
      printf "  Steady state (per-matmul, pipelined)     %8.1f us   <- with host quant/dequant\n",percall
      printf "    NPU matmul (our skel, on-DSP)          %8.1f      <- device, weight prefetch hidden\n",c
      printf "    Host compute (NEON)                    %8.1f      <- ARM quant+dequant (ref: SDKL fc_compare)\n",HOST+0
      print ""
      print "| method                        | relErr  | NPU us | host NEON us | per-call us | engine         |"
      print "|-------------------------------|---------|-------:|-------------:|------------:|----------------|"
      printf "| OURS cross-matmul (micro+DMA) | %-7s | %6.1f | %12.1f | %11.1f | micro+DMA skel |\n",REL,c,HOST+0,percall
      printf "| SDKL production (sdkl_npu_mm) | %-7s | %6.1f | %12.1f | %11.1f | NPU/HMX        |\n",REL,SDKL_PROD_NPU+0,HOST+0,SDKL_PROD_TOT+0
      printf "| SDKL on-DSP kernel (no RPC)   | %-7s | %6.1f | %12.1f | %11.1f | NPU/HMX        |\n",REL,s,HOST+0,sdkl_dsp_tot
      printf "| QNN (AI Engine Direct, doc06) | ~       | %6.1f | %12s | %11s | QNN            |\n",QNN+0,"(uint8 i/o)","-"
    }
  }'

cat <<'NOTE'

--------------------------------------------------------------------
Acceptance gate: the u8i4 rows above must reproduce doc 13 (medium 28.5us,
q_proj 56.3us, ffn_up 84.5us cross-matmul, +-10%) -- u8i4 runs first in every
loop in the .c file, same order as before this session's dtype
parametrisation, specifically so this comparison stays meaningful.

Apples-to-apples = the NPU-us column: our cross-matmul < QNN (69.9) < SDKL
on-DSP for u8i4 (~107). SDKL production's 270 NPU includes per-op FastRPC;
ours assumes a persistent skel (dspqueue batching), so no per-op RPC.
host NEON (76.7) is a REFERENCE constant from a separate SDKL fc_compare run
(same u8i4 quant recipe both sides); it does not change with dtype and is not
measured by this bench for u8i8 -- don't read it as a u8i8 number.
Override refs via env: HOST_NEON_US= SDKL_PROD_NPU_US= SDKL_PROD_TOT_US= QNN_FC_US=
--------------------------------------------------------------------
NOTE
