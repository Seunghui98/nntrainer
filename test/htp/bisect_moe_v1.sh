#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Builds the DSP skel AND the ARM test from whatever is checked out, pushes
# both, and runs the one test that says whether the batched MoE layer still
# matches the 64-call path it replaces.
#
# The point is that it builds BOTH. Bisecting this by hand put a HEAD skel
# next to an old test binary, and since every kernel change lives on the DSP
# side that combination answers nothing while looking like an answer. It also
# stops at the first failure, which a sequence of separate commands does not:
# test/htp/build.sh failing and the push running anyway is how the last
# attempt went.
#
#   ./test/htp/bisect_moe_v1.sh              # whatever is checked out
#   ./test/htp/bisect_moe_v1.sh <sha>        # checkout first
#
# Prints one line: bad_elems=N of 409600. Zero is a pass.
set -euo pipefail

# Same defaults as run_u8i4_layer_on_device.sh, so this needs no sourcing.
: "${HEXAGON_SDK_ROOT:=$HOME/workspace/Hexagon_SDK/6.4.0.2}"
: "${DEFAULT_HEXAGON_TOOLS_ROOT:=$HEXAGON_SDK_ROOT/tools/HEXAGON_Tools/19.0.04}"
: "${HEXKL_ROOT:=$HOME/workspace/hxkl-beta2/hexkl_addon}"
: "${HEXKL_SDK_VER:=6.4.0.2}"
: "${ANDROID_NDK:=$HOME/workspace/android-ndk-r26d}"
: "${DEVICE_TMP:=/data/local/tmp/htp_u8i4_layer_test}"
export HEXKL_ROOT HEXKL_SDK_VER HEXAGON_SDK_ROOT DEFAULT_HEXAGON_TOOLS_ROOT ANDROID_NDK

for v in HEXAGON_SDK_ROOT DEFAULT_HEXAGON_TOOLS_ROOT HEXKL_ROOT ANDROID_NDK; do
  [ -d "${!v}" ] || { echo "$v does not exist: ${!v}" >&2; exit 1; }
done
adb get-state >/dev/null 2>&1 || { echo "no device -- check adb devices" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

if [ $# -ge 1 ]; then
  echo "==> checkout $1"
  git checkout -q "$1"
fi
echo "==> $(git log --oneline -1)"

echo "==> skel"
HEXKL_ROOT="$HEXKL_ROOT" HEXKL_SDK_VER="$HEXKL_SDK_VER" bash "$HERE/build.sh"

echo "==> arm test"
( cd "$ROOT/test/jni" && "$ANDROID_NDK/ndk-build" \
    NDK_PROJECT_PATH=. NDK_APPLICATION_MK=./Application.mk \
    APP_BUILD_SCRIPT=./Android.mk NNTRAINER_ROOT="$ROOT" \
    HEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" unittest_hvx_mm_u8i4 >/dev/null )

echo "==> push + run"
adb shell "mkdir -p $DEVICE_TMP" >/dev/null
adb push "$HERE/build/libnntr_hvx_skel.so" \
         "$ROOT/test/jni/obj/local/arm64-v8a/unittest_hvx_mm_u8i4" \
         "$DEVICE_TMP/" >/dev/null
adb shell "cd $DEVICE_TMP && chmod +x unittest_hvx_mm_u8i4 && \
  LD_LIBRARY_PATH=$DEVICE_TMP ADSP_LIBRARY_PATH=$DEVICE_TMP \
  ./unittest_hvx_mm_u8i4 --gtest_filter='*MoeLayerMatchesTwoCall*'" \
  | grep -E "U8I4_FIELD|first at|OK \]|FAILED \]"
