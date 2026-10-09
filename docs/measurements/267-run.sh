#!/usr/bin/env bash
##
# @file    267-run.sh
# @brief   #267 sitting 1 (plan 267 S0): A = the 260 r2 sitting's E build,
#          B = A + L0 (OP_TIME split), C = B + NNTR_HTP_E2E_SPIN_US=12000,
#          D = B + L1 (src_bypass on the baked heap images); Gemma-4 26B
#          QS4CX, C 16, S25 R3CY205ZMND
#
# Reuses 260-e-run.sh's machinery (cool start, cell, the S1-ceiling gtest,
# the staged r2_E_* configs): its functions are sourced from R260 (a
# checkout of PR #265's branch, default ~/nntrainer-260doc). Every cell is
# an E cell (NNTR_HTP_E2E=1); the variant is the cell's tag.
#
# usage:
#   267-run.sh install <B app dir> <B skel> <D skel>
#                         push B's app + skel to s267b/, D's skel beside
#                         the same app to s267d/ (lock held by the caller)
#   267-run.sh wait            poll until the sitting lock is ours, then
#                              install if INSTALL="<app> <skelB> <skelD>"
#   267-run.sh one <log dir> <variant> <prompt> <G> [suffix] [extra env]
#                              one cell (lock held; C 16 only)
#   267-run.sh dumps <dir>...  md5 of each prefill MoE dump directory
#   267-run.sh release
#   267-run.sh sum <log dir>   the per-cell table (host only)
set -u -o pipefail
R260=${R260:-$HOME/nntrainer-260doc/docs/measurements}
# shellcheck disable=SC1090
source <(sed '/^case "${1:-}" in/,$d' "$R260/260-e-run.sh")
HERE=$R260 # 260-turn106.py and friends
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
CR=/data/local/tmp/nntrainer/causallm
declare -A BIN=([A]=$CR/s260r2 [B]=$CR/s267b [C]=$CR/s267b [D]=$CR/s267d)
declare -A XENV=([A]="" [B]="" [C]="NNTR_HTP_E2E_SPIN_US=12000" [D]="")
export SITTING_OWNER=${SITTING_OWNER:-r267@$(hostname)}

install() {
  local app=${1:?app dir} sb=${2:?B skel} sd=${3:?D skel} d f
  for d in s267b s267d; do
    $AD shell "mkdir -p $CR/$d && cd $CR/$d && cp $CR/s260r2/libc++_shared.so $CR/s260r2/libsdkl.so $CR/s260r2/unittest_hvx_two_sessions ."
    for f in jni/libs/arm64-v8a/nntrainer_causallm jni/libs/arm64-v8a/libcausallm_core.so \
      jni/obj/local/arm64-v8a/libnntrainer.so jni/obj/local/arm64-v8a/libccapi-nntrainer.so; do
      $AD push "$app/$f" "$CR/$d/" >/dev/null
    done
  done
  $AD push "$sb" "$CR/s267b/libnntr_hvx_skel.so" >/dev/null
  $AD push "$sd" "$CR/s267d/libnntr_hvx_skel.so" >/dev/null
  for f in unittest_hvx_mm_u8i4 unittest_hvx_fc; do
    $AD push "$ROOT/test/jni/obj/local/arm64-v8a/$f" "$CR/s267d/" >/dev/null
  done
  $AD shell "chmod 755 $CR/s267b/nntrainer_causallm $CR/s267d/nntrainer_causallm $CR/s267d/unittest_hvx_*"
  for d in s260r2 s267b s267d; do
    echo "== $d"
    $AD shell "cd $CR/$d && md5sum nntrainer_causallm libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libnntr_hvx_skel.so" | tr -d '\r'
  done
}

v() { # v <variant> <prompt> <G> [suffix] [extra env]
  cell "${BIN[$1]}" "$L" E "$2" "$3" "$1${4:+_$4}" "${XENV[$1]} ${5:-}"
}

cool() { # relaxed (coordinator, 2026-10-09): no other run, then battery
  # <= 32.0 C or a 5 minute cap; the start temperatures are logged
  local i soc bat z0
  while $AD shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done
  for i in $(seq 30); do
    read -r soc bat z0 <<<"$(temps)"
    [ "$bat" -le 320 ] && break
    sleep 10
  done
  echo "$soc $bat $z0"
}

wait_lock() { # poll until the sitting lock is ours, install once
  until (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "267 S0 D/B/C" 120); do
    sleep 60
  done
  # shellcheck disable=SC2086
  [ -n "${INSTALL:-}" ] && install $INSTALL
}

one() { # one <log dir> <variant> <prompt> <G> [suffix] [extra env]
  L=${1:?log dir}
  mkdir -p "$L/done"
  CLADDER=16 v "$2" "$3" "$4" "${5:-}" "${6:-}"
}

dumps() { # dumps <dir x> <dir y>: the prefill MoE dumps equal bit for bit
  local x
  for x in "$@"; do
    $AD shell "cd $x && ls moe_*.f32 | wc -l && md5sum manifest.txt moe_*.f32 | md5sum" | tr -d '\r' | tr '\n' ' '
    echo " $x"
  done
}

sum() { # host-only table; text compared to the 260 r2 sitting's E
  local L=${1:?log dir}
  python3 - "$L" "${REF:-$HOME/r260s-logs}" <<'PY'
import glob, os, re, sys
L, ref = sys.argv[1:]
def g(rx, s):
    m = re.findall(rx, s)
    return m[-1] if m else "-"
def text(s):
    m = "</Text><turn|>\n<|turn>model"
    i = s.find(m)
    j = s.find("=================[ LLM", i)
    return re.sub(r"\[(HTP|PPL)\][^\n]*\n", "", s[i + len(m):j]) if i >= 0 else None
print("| cell | prefill ms | prefill tok/s | decode tok/s (last 64) | tokens | text == 260 E | misses/token | miss_wait ms/token | wall ms/token | mhz | compute_mhz | ops/wall | FC ms | DENSE_FFN ms | LM_HEAD ms | ATTN_M1 ms | ROUTER ms | MOE ms (net of wait) | mapped MiB | heap_used_kib | peak RSS KiB | S1 ceiling | nll | zone0 start |")
print("|" + "---|" * 24)
for f in sorted(glob.glob(os.path.join(L, "E_*.log"))):
    b = os.path.basename(f)[:-4]
    s = open(f, errors="replace").read()
    pf = re.search(r"prefill: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    ge = re.search(r"generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    l64 = g(r"generation\(last 64\): \d+ tokens, \d+ ms, ([\d.]+) TPS", s)
    base = re.sub(r"_(A|B|C|D)(_.*)?$", "", b)
    rf = os.path.join(ref, base + ".log")
    t = text(s)
    same = "-"
    if "ppl" not in b and os.path.exists(rf) and t is not None:
        same = "yes" if t == text(open(rf, errors="replace").read()) else "**NO**"
    us = g(r"graph per-kind us/token:([^|]*)\|", s)
    kv = dict(re.findall(r"(\w+)=([\d.]+)", us)) if us != "-" else {}
    net = g(r"MOE=[\d.]+\(net of miss wait ([-\d.]+)\)", s)
    ms = lambda k: "%.2f" % (float(kv[k]) / 1000) if k in kv else "-"
    print("| %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s (%s) | %s | %s | %s | %s | %s | %s |" % (
        b, pf.group(2) if pf else "-", pf.group(3) if pf else "-",
        ge.group(3) if ge else "-", l64, ge.group(1) if ge else "-", same,
        g(r"misses/token=([\d.]+)", s),
        "%.1f" % (float(g(r"miss_wait_us/token=([\d.]+)", s)) / 1000) if "miss_wait_us" in s else "-",
        g(r"wall_ms/token=([\d.]+)", s), g(r" mhz=(\d+)", s), g(r"compute_mhz=(\d+)", s),
        g(r"ops/wall=([\d.]+)", s), ms("FC"), ms("DENSE_FFN"), ms("LM_HEAD"),
        ms("ATTN_M1"), ms("ROUTER_TOPK"), ms("MOE"),
        "%.2f" % (float(net) / 1000) if net != "-" else "-",
        g(r"mapped_mib=([\d.]+)", s), g(r"heap_used_kib=(\d+)", s),
        g(r"Max RSS \(KiB\): (\d+)", s), g(r"S1_CEILING (\S+)", s),
        g(r"nll/token=([\d.]+)", s),
        (re.findall(r"start soc=\d+ bat=\d+ zone0=(\d+)", s) or ["-"])[0]))
PY
  cat "$L/dumps.txt" 2>/dev/null
}

case "${1:-}" in
install) shift; install "$@" ;;
wait) wait_lock ;;
one) shift; one "$@" ;;
dumps) shift; dumps "$@" ;;
release) (cd "$ROOT" && tools/htp/sitting_lock.sh release $SER) ;;
sum) shift; sum "$@" ;;
*) sed -n '2,25p' "$0"; exit 1 ;;
esac
