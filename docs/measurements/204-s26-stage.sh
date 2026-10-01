#!/usr/bin/env bash
##
# @file    204-s26-stage.sh
# @brief   Stages plan 204's S26 re-baseline sitting from a built checkout
#          (handoff docs/measurements/204-s26-rebaseline.md)
#
# Copies the app set, the v81 skel, the five device gtests, the evictor
# (built here with the NDK), the prompt and the runner into the stage
# directory and writes md5.txt, which the runner checks on both ends.
# Refuses a skel whose ELF flags do not say v81 (both arches share the name).
# Usage: bash docs/measurements/204-s26-stage.sh [checkout] [stage dir]
set -eo pipefail
R=${1:-/home/j2z0-lee/nntrainer}
W=${2:-/local/mnt/workspace/htp_moe/204/s26}
NDK=${ANDROID_NDK:-$HOME/android-ndk-r30}
readelf -h "$R"/test/htp/build/libnntr_hvx_skel.so | grep -q 'Flags:.*0x81\b' ||
  { echo "the skel in $R/test/htp/build is not v81 (HEX_ARCH=v81 ./test/htp/build.sh)" >&2; exit 1; }
A=$W/app
rm -rf "$A"; mkdir -p "$A"
cp "$R"/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} \
   "$R"/Applications/CausalLM/jni/obj/local/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so} \
   "$R"/test/htp/build/libnntr_hvx_skel.so \
   "$R"/test/jni/obj/local/arm64-v8a/unittest_hvx_{mm_u8i4,softmax,attn,fc,two_sessions} "$A"/
cp "$NDK"/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so "$A"/
cp "${HEXKL_ROOT:-$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon}"/lib/6.4.0.1/armv8_android26/libsdkl.so "$A"/
"$NDK"/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang -O2 -Wall \
  -o "$A"/page_cache_evict "$R"/tools/htp/page_cache_evict.c
cp "$R"/docs/measurements/77-prompt512.txt "$A"/prompt512.txt
cp "$R"/docs/measurements/204-s26-run.sh "$W"/run_s26.sh
(cd "$W" && { find app -type f | sort | xargs md5sum; md5sum run_s26.sh; } > md5.txt)
echo "staged from $R @ $(git -C "$R" rev-parse --short=9 HEAD) "
cat "$W"/md5.txt
