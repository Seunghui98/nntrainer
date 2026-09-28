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

# [#141] dspqueue.h: the types and the entries HtpDspqApi takes decltype of.
cat > "$STUB/dspqueue.h" <<'H'
#pragma once
#include <stdint.h>
typedef int AEEResult;
#define AEE_EWOULDBLOCK 516
#define DSPQUEUE_TIMEOUT_NONE 0xffffffff
enum dspqueue_buffer_flags {
  DSPQUEUE_BUFFER_FLAG_REF = 4, DSPQUEUE_BUFFER_FLAG_DEREF = 8,
  DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER = 0x10,
  DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT = 0x80
};
enum dspqueue_stat { DSPQUEUE_STAT_SIGNALING_PERF = 7 };
struct dspqueue;
typedef struct dspqueue *dspqueue_t;
struct dspqueue_buffer {
  uint32_t fd, size, offset, flags;
  union { void *ptr; uint64_t address; };
};
typedef void (*dspqueue_callback_t)(dspqueue_t, AEEResult, void *);
extern "C" {
AEEResult dspqueue_create(int, uint32_t, uint32_t, uint32_t,
                          dspqueue_callback_t, dspqueue_callback_t, void *,
                          dspqueue_t *);
AEEResult dspqueue_close(dspqueue_t);
AEEResult dspqueue_export(dspqueue_t, uint64_t *);
AEEResult dspqueue_write(dspqueue_t, uint32_t, uint32_t,
                         struct dspqueue_buffer *, uint32_t, const uint8_t *,
                         uint32_t);
AEEResult dspqueue_read_noblock(dspqueue_t, uint32_t *, uint32_t, uint32_t *,
                                struct dspqueue_buffer *, uint32_t, uint32_t *,
                                uint8_t *);
AEEResult dspqueue_read(dspqueue_t, uint32_t *, uint32_t, uint32_t *,
                        struct dspqueue_buffer *, uint32_t, uint32_t *,
                        uint8_t *, uint32_t);
AEEResult dspqueue_get_stat(dspqueue_t, enum dspqueue_stat, uint64_t *);
}
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
