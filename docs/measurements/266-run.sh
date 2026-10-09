#!/usr/bin/env bash
##
# @file    266-run.sh
# @brief   #266 S1 sitting: the flash miss readers against the 260 r2 E
#          install, Gemma-4 26B QS4CX file, S25 R3CY205ZMND
#
# Plan: docs/plans/266-flash-miss-path.md section 4 S1. Every cell is E (the
# one-PD decode, NNTR_HTP_E2E=1, C = 16, the r2_E_* configs of the 260 r2
# sitting); the variants are the install and one knob:
#   A   the 260 r2 install s260r2/ (htp_decode @ 3897963d8), unchanged
#   B   the S1 install s266/, default knobs (4 miss readers); its G512 cells
#       write NNTR_HTP_ROUTE_LOG
#   D   s266/ with NNTR_MOE_MISS_READERS=1 (whole-weight reads, one at a
#       time: the request shape without the parallelism)
# Cells, cool starts, the S1-ceiling column and the logs come from
# 260-e-run.sh (sourced). A first and last.
#
# usage:
#   266-run.sh run <log dir>   the sitting (takes and releases the lock)
#   266-run.sh quick <log dir> B p1024 G512, then B p512 G64 PPL (the cut
#                              plan, user 2026-10-09; COOL_QUICK)
#   266-run.sh gate <log dir>  per cell: decode tok/s, the pool line, text
#                              and nll against the 260 r2 sitting's E cell
#                              of the same prompt and G (REF=<its log dir>)
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=260-e-run.sh
source "$HERE/260-e-run.sh"
BA=/data/local/tmp/nntrainer/causallm/s260r2
BS=/data/local/tmp/nntrainer/causallm/s266
RL=/data/local/tmp/nntrainer/r266

run266() {
  local L=${1:?log dir} p g
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "266 S1 A/B/D" 420) || exit 1
  trap '(cd "$ROOT" && tools/htp/sitting_lock.sh release $SER)' EXIT
  echo "=== 266 S1 sitting $(date '+%F %T %Z') uptime=$($AD shell cat /proc/uptime | tr -d '\r')"
  export CLADDER=16 # C stays 16 (user 2026-10-09): no ladder
  for pg in "512 512" "1024 512" "512 64" "1024 64" "512 1024" "1024 1024"; do
    read -r p g <<<"$pg"
    cell $BA "$L" E "$p" "$g" A
    if [ "$g" = 512 ]; then
      cell $BS "$L" E "$p" "$g" B "NNTR_HTP_ROUTE_LOG=$RL/route_p${p}_g${g}.txt"
    else
      cell $BS "$L" E "$p" "$g" B
    fi
    cell $BS "$L" E "$p" "$g" D NNTR_MOE_MISS_READERS=1
  done
  for p in 512 1024; do # prompt nll (the prefill rows; S1 is decode only)
    cell $BS "$L" E "$p" 64 B_ppl NNTR_PPL=1
  done
  cell $BA "$L" E 512 512 A_last # A first and last (drift)
  $AD shell "ls -la $RL/route_*" | tr -d '\r'
  echo "=== done $(date '+%F %T %Z')"
}

quick() { # user 2026-10-09: the minimum, in order; the 260 r2 sitting's E
  # rows stand for A. B p1024 G512, then (if B's text == E) the PPL cell;
  # B / D p512 G512 ran in the first (stopped) grid. Cool: COOL_QUICK.
  local L=${1:?log dir}
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "266 S1 quick" 30) || exit 1
  trap '(cd "$ROOT" && tools/htp/sitting_lock.sh release $SER)' EXIT
  export CLADDER=16 COOL_QUICK=1
  echo "=== 266 S1 quick $(date '+%F %T %Z')"
  cell $BS "$L" E 1024 512 B "NNTR_HTP_ROUTE_LOG=$RL/route_p1024_g512.txt"
  cell $BS "$L" E 512 64 B_ppl NNTR_PPL=1
  $AD shell "ls -la $RL/route_*" | tr -d '\r'
  echo "=== done $(date '+%F %T %Z')"
}

gate() {
  local L=${1:?log dir}
  python3 - "$L" "${REF:?REF=<260 r2 log dir>}" <<'PY'
import glob, os, re, sys
L, ref = sys.argv[1:]
def gen(s):  # the generated text, as 260-turn106.py cuts it
    m = "</Text><turn|>\n<|turn>model"
    i = s.find(m)
    j = s.find("=================[ LLM", i)
    return re.sub(r"\[(HTP|PPL)\][^\n]*\n", "", s[i + len(m):j]) if i >= 0 else None
def g(rx, s):
    m = re.findall(rx, s)
    return m[-1] if m else "-"
print("| cell | decode tok/s | tokens | misses/token | miss_wait ms/token | rounds | arm_ms/round | pgpgin MiB/miss | text == 260 r2 E | nll | ref nll | peak RSS KiB | S1 ceiling | zone0 start |")
print("|" + "---|" * 14)
for f in sorted(glob.glob(os.path.join(L, "E_*.log"))):
    b = os.path.basename(f)[:-4]
    if b.startswith("ceil_") or b.endswith("fail"):
        continue
    s = open(f, errors="replace").read()
    base = re.sub(r"_(A|B|D)(_last|_ppl)?$", "", b) + ("_ppl" if b.endswith("_ppl") else "")
    rf = os.path.join(ref, base + ".log")
    r = open(rf, errors="replace").read() if os.path.exists(rf) else ""
    ge = re.search(r"generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    pool = re.findall(r"pool misses=(\d+) misses/token=([\d.]+) miss_wait_us/token=([\d.]+) rounds=(\d+) arm_ms/round=([\d.]+) pgpgin_mib=([\d.]+)", s)
    pl = pool[-1] if pool else None
    t, tr = gen(s), gen(r) if r else None
    same = "-" if t is None or tr is None else ("yes" if t == tr else "**NO**")
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
        b, ge.group(3) if ge else "-", ge.group(1) if ge else "-",
        pl[1] if pl else "-", "%.1f" % (float(pl[2]) / 1000) if pl else "-",
        pl[3] if pl else "-", pl[4] if pl else "-",
        "%.2f" % (float(pl[5]) / int(pl[0])) if pl and int(pl[0]) else "-",
        same, g(r"nll/token=([\d.]+)", s), g(r"nll/token=([\d.]+)", r),
        g(r"Max RSS \(KiB\): (\d+)", s), g(r"S1_CEILING (\S+)", s),
        (re.findall(r"start soc=\d+ bat=\d+ zone0=(\d+)", s) or ["-"])[0]))
PY
}

case "${1:-}" in
run) shift; run266 "$@" ;;
quick) shift; quick "$@" ;;
gate) shift; gate "$@" ;;
*) sed -n '2,24p' "$0"; exit 1 ;;
esac
