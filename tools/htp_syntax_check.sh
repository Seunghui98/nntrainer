#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Type-checks the HTP backend's host sources without a Hexagon SDK.
#
# htp_compute_ops.cpp compiles only under -Denable-htp, which needs the SDK
# and the qaic-generated stub, so a machine without either cannot build it at
# all -- and two mistakes in it have reached a device build that way, each
# costing a full cycle. This gets a compiler over the file with stand-ins for
# the two headers that are otherwise unavailable.
#
# What it checks: everything about this tree's own code -- declaration order,
# member names, types, overload resolution against ComputeOps.
#
# What it does NOT check: the FastRPC signatures. nntr_hvx.h is generated from
# nntr_hvx.idl by qaic, and the stand-in here declares every entry point as
# variadic, so a call with the wrong argument list still passes. Only a real
# HTP build catches that.
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
SRC="$ROOT/nntrainer/tensor/htp_backend/htp_compute_ops.cpp"
STUB="$(mktemp -d)"
trap 'rm -rf "$STUB"' EXIT

cat > "$STUB/remote.h" <<'H'
#pragma once
#include <stdint.h>
#include <stddef.h>
typedef uint64_t remote_handle64;
enum { CDSP_DOMAIN_ID = 3 };
enum fastrpc_map_flags {
  FASTRPC_MAP_STATIC = 0, FASTRPC_MAP_RESERVED, FASTRPC_MAP_FD,
  FASTRPC_MAP_FD_DELAYED
};
struct remote_rpc_control_unsigned_module { int domain; int enable; };
enum { DSPRPC_CONTROL_UNSIGNED_MODULE = 1 };
extern "C" int remote_session_control(uint32_t, void *, uint32_t);
H

cat > "$STUB/AEEStdErr.h" <<'H'
#pragma once
#define AEE_EOFFSET 0
#define AEE_SUCCESS 0
#define AEE_EBADPARM 14
#define AEE_ENOMEMORY 2
#define AEE_EBADSTATE 13
#define AEE_EUNSUPPORTED 20
H

# Variadic on purpose: see the note above about what this cannot check.
{
  echo '#pragma once'
  echo '#include <remote.h>'
  echo '#include <AEEStdErr.h>'
  echo 'extern "C" {'
  grep -ohE 'nntr_hvx_[a-z0-9_]+' "$SRC" | sort -u | sed 's/^/int /; s/$/(...);/'
  echo '}'
} > "$STUB/nntr_hvx.h"

INC=(nntrainer api api/ccapi/include nntrainer/tensor nntrainer/tensor/htp_backend
     nntrainer/tensor/cpu_backend nntrainer/tensor/cpu_backend/fallback
     nntrainer/tensor/cpu_backend/x86 nntrainer/utils nntrainer/layers
     nntrainer/graph nntrainer/models nntrainer/compiler nntrainer/optimizers
     nntrainer/dataset nntrainer/tensor/cpu_backend/ggml_interface
     nntrainer/tensor/cpu_backend/ggml_interface/nntr_ggml_impl
     nntrainer/tensor/cpu_backend/cblas_interface)
ARGS=()
for d in "${INC[@]}"; do ARGS+=("-I$ROOT/$d"); done

"${CXX:-g++}" -std=c++17 -fsyntax-only -DENABLE_HEXKL=1 -DMIN_CPP_VERSION=201703L \
  "${ARGS[@]}" -I"$STUB" "$SRC"
echo "htp_compute_ops.cpp: type-checks (FastRPC signatures NOT covered)"
