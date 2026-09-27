#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
##
# @file    env.sh
# @brief   Environment for HTP builds: the Ubuntu workstation or the dev container
# @author  dlwlzzero <dlwlzzero@gmail.com>
#
# Contract: docs/plans/0001-htp-moe-decode-agent-system.md §4.1.
# Source this before test/htp/build.sh, build_android.sh --htp or the
# host build:
#
#     source tools/htp/env.sh
#
# Two hosts are known. On the Ubuntu workstation the SDK lives under
# /local/mnt/workspace and HexKL under ~/Qualcomm. Inside the dev container
# (tools/docker/run.sh, the Mac client) run.sh mounts the SDK versions at
# /opt/qcom/Hexagon_SDK/<ver> and the HexKL package at /opt/qcom/hexkl_addon,
# and the image ships the NDK at $ANDROID_NDK; the entrypoint has already
# sourced setup_sdk_env.source of the newest SDK. Both layouts of the HexKL
# package are accepted: lib/<sdk version>/hexagon_toolv19_<arch>/ and the
# flat lib/hexagon_toolv19_<arch>/ (HEXKL_SDK_VER=.).
#
# Overrides: HEXAGON_SDK_ROOT, HEXKL_ROOT (the HexKL package root that holds
# lib/...), HEXKL_SDK_VER (the lib/ subdirectory to use, "." for the flat
# layout), ANDROID_NDK, NNTR_MODEL_DIR, HEX_ARCH (v79, the S25 Ultra).

_nntr_in_container=0
[ -d /opt/qcom/Hexagon_SDK ] && [ -f /work/tools/htp/env.sh ] && _nntr_in_container=1

if [ -z "${HEXAGON_SDK_ROOT:-}" ] || [ ! -f "${HEXAGON_SDK_ROOT}/setup_sdk_env.source" ]; then
  if [ "$_nntr_in_container" = 1 ]; then
    _nntr_sdk="$(ls -d /opt/qcom/Hexagon_SDK/*/ 2>/dev/null | sort -V | tail -1)"
    _nntr_sdk="${_nntr_sdk%/}"
  else
    _nntr_sdk="/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1"
  fi
else
  _nntr_sdk="$HEXAGON_SDK_ROOT"
fi
if [ ! -f "$_nntr_sdk/setup_sdk_env.source" ]; then
  echo "env.sh: no SDK at $_nntr_sdk" >&2
  unset _nntr_sdk _nntr_in_container
  return 1 2>/dev/null || exit 1
fi
# setup_sdk_env.source returns at once when HEXAGON_SDK_ROOT is already set.
unset HEXAGON_SDK_ROOT
# shellcheck disable=SC1091
source "$_nntr_sdk/setup_sdk_env.source" >/dev/null
unset _nntr_sdk

if [ -z "${HEXKL_ROOT:-}" ]; then
  if [ "$_nntr_in_container" = 1 ] && [ -d /opt/qcom/hexkl_addon/lib ]; then
    HEXKL_ROOT=/opt/qcom/hexkl_addon
  else
    HEXKL_ROOT="$HOME/Qualcomm/hexkl-1.0-beta.2/hexkl_addon"
  fi
fi
export HEXKL_ROOT
export HEX_ARCH="${HEX_ARCH:-v79}"
if [ -z "${HEXKL_SDK_VER:-}" ]; then
  HEXKL_SDK_VER="$(basename "$HEXAGON_SDK_ROOT")"
  if [ ! -d "$HEXKL_ROOT/lib/$HEXKL_SDK_VER" ] && [ -d "$HEXKL_ROOT/lib/hexagon_toolv19_$HEX_ARCH" ]; then
    HEXKL_SDK_VER=.   # flat package layout (the hvx_impl-style view)
  fi
fi
export HEXKL_SDK_VER
if [ "$_nntr_in_container" = 1 ]; then
  export ANDROID_NDK="${ANDROID_NDK:-/opt/android-ndk-r26d}"
  export NNTR_MODEL_DIR="${NNTR_MODEL_DIR:-/model}"
  # qaic reads the IDL through the locale; the image has none set.
  export LANG="${LANG:-C.UTF-8}" LC_ALL="${LC_ALL:-C.UTF-8}"
else
  export ANDROID_NDK="${ANDROID_NDK:-$HOME/android-ndk-r30}"
  export NNTR_MODEL_DIR="${NNTR_MODEL_DIR:-/local/mnt/workspace/models/lfm2.5-8b-a1b}"
fi
export ANDROID_NDK_HOME="$ANDROID_NDK"
for _f in "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/hexagon_toolv19_$HEX_ARCH/libhexkl_micro.a" \
          "$HEXKL_ROOT/lib/$HEXKL_SDK_VER/armv8_android26/libsdkl.so" \
          "$ANDROID_NDK/ndk-build"; do
  [ -e "$_f" ] || echo "env.sh: missing $_f" >&2
done
unset _f
# Tools/bin (hexagon-clang), the NDK, ~/.cargo/bin (cargo for the tokenizer
# library of the Android app) and ~/.local/bin (clang-format-14)
for _d in "$DEFAULT_HEXAGON_TOOLS_ROOT/Tools/bin" "$ANDROID_NDK" "$HOME/.cargo/bin" "$HOME/.local/bin"; do
  case ":$PATH:" in *":$_d:"*) ;; *) export PATH="$_d:$PATH" ;; esac
done
unset _d _nntr_in_container

echo "htp env: SDK $HEXAGON_SDK_ROOT ($DEFAULT_TOOLS_VARIANT), HexKL $HEXKL_ROOT/lib/$HEXKL_SDK_VER, NDK $ANDROID_NDK, model $NNTR_MODEL_DIR, arch $HEX_ARCH"
