#!/usr/bin/env bash
##
# @file    260-e-run.sh
# @brief   #260 revision 2 sitting: E (#4415 prefill + our one-PD decode) and
#          A (#4415's hybrid on the same build) on the standard 512 / 1024
#          prompts, Gemma-4 26B QS4CX file, S25 R3CY205ZMND
#
# Plan: docs/plans/260-4415-prefill-port.md section 7 (7.2, 7.3, 7.5).
#   E    NNTR_HTP_E2E=1, lmhead_engine cpu (the head placed once, as Q4M1 for
#        the token), attention_kv_dtype unset (E never runs on q8), C = 16
#        (moe_cache_experts; the pool is the LRU's slot set).
#   A    the same build without NNTR_HTP_E2E: #4415's prefill and hybrid
#        decode, lmhead_engine htp, attention_kv_dtype unset.
#   Aq8  A with attention_kv_dtype q8, G 64 only (KV-format control).
#   Ap   A on the pure #4415 @ 41bd65e04 install (s260p3), one cell
#        (p512 G512), to show the merged build's A path is #4415's own:
#        same text and nll, prefill within 5 %. If it is not, the A rows of
#        the sitting are read from Ap and the doc says so.
#
# usage:
#   260-e-run.sh stage                   build the cell configs from the
#                                        delivered nntr_config.json and the
#                                        standard prompts, push them to
#                                        /data/local/tmp/nntrainer/s260cfg/r2_*
#                                        (no model run; no lock needed)
#   260-e-run.sh run <bin dir> <log dir> the sitting (takes and releases the
#                                        lock). <bin dir> is the merged
#                                        build's install dir on the phone and
#                                        holds unittest_hvx_two_sessions
#   260-e-run.sh cx <bin dir> <log dir> <C>...
#                                        E p512 G512 at each C (no ladder;
#                                        stops at the first C that cannot
#                                        generate), the miss curve of the pool
#   260-e-run.sh sum <log dir>           the per-cell table, text, first
#                                        <turn|> and loop onset (host only)
# Resumable per cell (<log dir>/done/<cell>). If an E cell cannot load, it is
# retried with NNTR_MOE_CACHE_EXPERTS=12, then 8 (the C ladder, plan 7.5);
# the C that ran is in the cell's log and the table.
set -u -o pipefail
SER=R3CY205ZMND
AD="adb -s $SER"
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=/local/mnt/workspace/models/gemma4_26b_qs4cx
DM=/data/local/tmp/nntrainer/gemma4_26b_qs4cx
DC=/data/local/tmp/nntrainer/s260cfg
PURE=/data/local/tmp/nntrainer/causallm/s260p3
TOK=$MODEL/tokenizer.json

stage() {
  local out
  out=$(mktemp -d)
  python3 - "$MODEL/nntr_config.json" "$HERE/260-prompt512.txt" \
    "$HERE/260-prompt1024.txt" "$out" "$DM" <<'PY'
import copy, json, os, sys
base_f, p512, p1024, out, dm = sys.argv[1:]
base = json.load(open(base_f))
base.update(skip_prefill=False,
            model_file_name=dm + "/nntr_gemma4_qs4cx_fc_arm.bin",
            tokenizer_file=dm + "/tokenizer.json")
base.pop("attention_kv_dtype", None)  # plan 7.3: unset in the main grid
prompts = {512: open(p512).read(), 1024: open(p1024).read()}
cells = []
for v in ("E", "A"):
    for p in (512, 1024):
        for g in (64, 512, 1024):
            cells.append((v, p, g))
cells += [("Aq8", 512, 64), ("Aq8", 1024, 64)]
for v, p, g in cells:
    d = copy.deepcopy(base)
    d["sample_input"] = prompts[p]
    d["num_to_generate"] = g
    # doc 57 7.5: a prompt as long as init_seq_len takes another branch
    d["init_seq_len"] = 2048 if p == 1024 else 1024
    d["lmhead_engine"] = "cpu" if v == "E" else "htp"
    if v == "Aq8":
        d["attention_kv_dtype"] = "q8"
    name = "r2_%s_p%d_g%d" % (v, p, g)
    os.makedirs(os.path.join(out, name))
    json.dump(d, open(os.path.join(out, name, "nntr_config.json"), "w"),
              indent=2, ensure_ascii=False)
PY
  for d in "$out"/r2_*; do
    n=$(basename "$d")
    $AD shell mkdir -p "$DC/$n" >/dev/null
    $AD push "$d/nntr_config.json" "$DC/$n/" >/dev/null
    $AD shell "cd $DC/$n && ln -sf $DM/config.json config.json && \
      ln -sf $DM/generation_config.json generation_config.json && \
      ln -sf $DM/tokenizer.json tokenizer.json"
  done
  (cd "$out" && md5sum r2_*/nntr_config.json)
  $AD shell "cd $DC && md5sum r2_*/nntr_config.json" | tr -d '\r' >"$out/device.md5"
  (cd "$out" && md5sum r2_*/nntr_config.json | diff - device.md5 >/dev/null) &&
    echo "STAGED: device md5 == local for $(ls -d "$out"/r2_* | wc -l) configs"
  rm -rf "$out"
}

temps() { # hottest cpu*/nsp* zone, battery (0.1 C), zone0 (m C)
  $AD shell 'm=0; for z in /sys/class/thermal/thermal_zone*; do case "$(cat $z/type)" in cpu*|nsp*) t=$(cat $z/temp); [ "$t" -gt "$m" ] && m=$t;; esac; done; b=$(dumpsys battery | grep "  temperature" | tr -dc 0-9); z0=$(cat /sys/class/thermal/thermal_zone0/temp); echo "$m $b $z0"' | tr -d '\r'
}
cool() { # doc 57 7.5 + #234: no other run, 180 s, then cpu/nsp <= 38 C,
  # battery <= 30.0 C, zone0 <= 35 C (20 min cap; the temps are logged)
  local i soc bat z0
  while $AD shell 'ps -A' | grep -q 'nntrainer_causall[m]'; do sleep 10; done
  # COOL_QUICK=1 (#266, user 2026-10-09): battery <= 32.0 C only, 5 min cap
  [ "${COOL_QUICK:-0}" = 1 ] || sleep "${COOL_S:-180}"
  for i in $(seq "$([ "${COOL_QUICK:-0}" = 1 ] && echo 30 || echo 120)"); do
    read -r soc bat z0 <<<"$(temps)"
    if [ "${COOL_QUICK:-0}" = 1 ]; then [ "$bat" -le 320 ] && break
    elif [ "$soc" -le 38000 ] && [ "$bat" -le 300 ] && [ "$z0" -le 35000 ]; then break; fi
    sleep 10
  done
  echo "$soc $bat $z0"
}
vm() { $AD shell "grep -E '^(pgpgin|workingset_refault_file) ' /proc/vmstat" | tr -d '\r' | awk '{printf "%s=%s ", $1, $2}'; }
ceil() { # S1's mmap ceiling after a run (#211); n/a without the gtest
  local b=$1 n=$2 L=$3 c
  if ! $AD shell "[ -x $b/unittest_hvx_two_sessions ]"; then
    echo "S1_CEILING n/a (no unittest_hvx_two_sessions in $b)" >>"$L/$n.log"; return 0
  fi
  $AD shell "cd $b && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. ./unittest_hvx_two_sessions \
    --gtest_filter=TwoSessions.S1Ceiling" >"$L/ceil_$n.log" 2>&1
  c=$(grep -o 'CEILING s1_mmap_mib=[0-9]*' "$L/ceil_$n.log" | cut -d= -f2)
  echo "S1_CEILING ${c:-?} MiB" >>"$L/$n.log"
  [ "${c:-0}" -ge 3840 ] || echo "WARN: S1 ceiling ${c:-?} MiB after $n (leak?)"
}

cell() { # cell <bin dir> <log dir> <variant> <prompt> <G> [tag] [extra env]
  local b=$1 L=$2 v=$3 p=$4 g=$5 tag=${6:-} extra=${7:-}
  local cfg="r2_${v}_p${p}_g${g}" n c e soc bat z0 v0 v1 rc
  [ "$v" = Ap ] && cfg="r2_A_p${p}_g${g}"
  n="${v}_p${p}_g${g}${tag:+_$tag}"
  [ -f "$L/done/$n" ] && { echo "$n: done earlier"; return 0; }
  for c in ${CLADDER:-16 12 8}; do
    e="NNTR_MOE_CACHE_EXPERTS=$c $extra"
    [ "$v" = E ] && e="NNTR_HTP_E2E=1 $e"
    read -r soc bat z0 <<<"$(cool)"
    v0=$(vm)
    {
      echo "CELL $n cfg=$cfg bin=$b env=[$e] start soc=$soc bat=$bat zone0=$z0 $(date -Is)"
      $AD shell "cd $b && md5sum nntrainer_causallm libcausallm_core.so libnntrainer.so libccapi-nntrainer.so libnntr_hvx_skel.so; md5sum $DC/$cfg/nntr_config.json"
      $AD logcat -c
      $AD shell "cd $b && env LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 $e \
        /system/bin/time -v ./nntrainer_causallm $DC/$cfg" 2>&1
      echo "RC=$?"
      read -r soc bat z0 <<<"$(temps)"
      echo "END soc=$soc bat=$bat zone0=$z0 $(date -Is)"
      echo "vm_before $v0"; echo "vm_after $(vm)"
      echo "== logcat (mha_core / HTP)"
      $AD logcat -d | grep -aE 'mha_core|HTP|hexkl|nntr_hvx' | tail -300
    } >"$L/$n.log" 2>&1
    ceil "$b" "$n" "$L"
    if grep -q '^generation:' "$L/$n.log"; then
      touch "$L/done/$n"
      echo "$n (C=$c): $(grep -h -E '^(prefill|generation):' "$L/$n.log" | grep -o '[0-9.]* TPS' | tr '\n' ' ')$(grep -aho 'calls/token=[0-9.]*' "$L/$n.log" | tail -1) $(grep -aho 'nll/token=[0-9.]*' "$L/$n.log") $(grep -ao 'Max RSS (KiB): [0-9]*' "$L/$n.log")"
      return 0
    fi
    echo "$n cannot generate at C=$c: $(grep -m1 -aE 'FATAL|Abort|ERROR|AEE_|what\(\)' "$L/$n.log" | cut -c1-200)"
    mv "$L/$n.log" "$L/$n.C$c.fail.log"
    [ "$v" = E ] || break # the C ladder is E's (plan 7.5)
  done
  touch "$L/done/$n"
  return 1
}

run() {
  local B=${1:?bin dir} L=${2:?log dir} p g
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "260 r2 E/A grid" 180) || exit 1
  trap '(cd "$ROOT" && tools/htp/sitting_lock.sh release $SER)' EXIT
  echo "=== 260 r2 sitting $(date '+%F %T %Z') bin=$B uptime=$($AD shell cat /proc/uptime | tr -d '\r')"
  # first deliverable first (plan 7.6 step 3): E p512 G512, then A
  cell "$B" "$L" E 512 512
  cell "$B" "$L" A 512 512
  cell "$PURE" "$L" Ap 512 512 # A on the pure #4415 install
  for p in 512 1024; do
    for g in 64 512 1024; do
      cell "$B" "$L" E $p $g
      cell "$B" "$L" A $p $g
    done
  done
  cell "$B" "$L" Aq8 512 64
  cell "$B" "$L" Aq8 1024 64
  for p in 512 1024; do # per-kind lines, one G512 run per prompt
    cell "$B" "$L" E $p 512 optime NNTR_OP_TIME=1
    cell "$B" "$L" A $p 512 optime NNTR_OP_TIME=1
  done
  for p in 512 1024; do # prompt nll (lm_head scores every prefill row)
    cell "$B" "$L" E $p 64 ppl NNTR_PPL=1
    cell "$B" "$L" A $p 64 ppl NNTR_PPL=1
  done
  cell "$B" "$L" A 512 64 last # A first and last (drift)
  echo "=== done $(date '+%F %T %Z')"
}

cx() { # cx <bin dir> <log dir> <C>... : E p512 G512 per C, lock held
  local B=${1:?bin dir} L=${2:?log dir} c
  shift 2
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  (cd "$ROOT" && tools/htp/sitting_lock.sh take $SER "260 r2 C sweep" 90) || exit 1
  trap '(cd "$ROOT" && tools/htp/sitting_lock.sh release $SER)' EXIT
  for c in "$@"; do
    CLADDER=$c cell "$B" "$L" E 512 512 "c$c" || { echo "stop: C=$c cannot generate"; break; }
  done
  echo "=== cx done $(date '+%F %T %Z')"
}

sum() { # host-only table
  local L=${1:?log dir}
  python3 - "$L" "$TOK" "$HERE" <<'PY'
import glob, os, re, subprocess, sys
L, tok, here = sys.argv[1:]
def g(rx, s, k=1):
    m = re.findall(rx, s)
    return m[-1] if m else "-"
print("| cell | C | prefill | prefill tok/s | decode tok/s | calls/token | router pcyc/op | mhz | id_checked | token_ms | mapped | heap_used_kib | arena MiB | peak RSS KiB | S1 ceiling | pgpgin / refault (delta) | nll | zone0 start | first <turn\\|> | loop onset |")
print("|" + "---|" * 20)
for f in sorted(glob.glob(os.path.join(L, "*.log"))):
    b = os.path.basename(f)
    if b.startswith("ceil_") or b.endswith("fail.log"):
        continue
    s = open(f, errors="replace").read()
    pf = re.search(r"prefill: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    ge = re.search(r"generation: (\d+) tokens, (\d+) ms, ([\d.]+) TPS", s)
    vb = dict(re.findall(r"(\w+)=(\d+)", g(r"vm_before (.*)", s)))
    va = dict(re.findall(r"(\w+)=(\d+)", g(r"vm_after (.*)", s)))
    dv = lambda k: str(int(va.get(k, 0)) - int(vb.get(k, 0))) if k in va and k in vb else "-"
    t106 = subprocess.run(["python3", os.path.join(here, "260-turn106.py"), tok, f],
                          capture_output=True, text=True).stdout
    lp = subprocess.run(["python3", os.path.join(here, "260-loopstart.py"), tok, f],
                        capture_output=True, text=True).stdout
    print("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %s |" % (
        b[:-4], g(r"NNTR_MOE_CACHE_EXPERTS=(\d+)", s),
        "%s tok, %s ms" % (pf.group(1), pf.group(2)) if pf else "-",
        pf.group(3) if pf else "-", ge.group(3) if ge else "-",
        g(r"calls/token=([\d.]+)", s), g(r"pcyc/op=([\d.]+)", s), g(r"mhz=([\d.]+)", s),
        g(r"id_checked=(\d+)", s), g(r"token_ms=([\d.]+)", s),
        g(r"mapped(?:_mib)?=([\d.]+)", s), g(r"heap_used_kib=(\d+)", s),
        g(r"arena: chunks unmapped \d+/\d+ mib=(\d+)", s), g(r"Max RSS \(KiB\): (\d+)", s),
        g(r"S1_CEILING (\S+)", s), dv("pgpgin"), dv("workingset_refault_file"),
        g(r"nll/token=([\d.]+)", s),
        (re.findall(r"start soc=\d+ bat=\d+ zone0=(\d+)", s) or ["-"])[0],
        g(r"first_turn_end_at=(\S+)", t106).replace("None", "none"),
        g(r"third10_at=(\S+)", lp).replace("None", "none")))
print()
for f in sorted(glob.glob(os.path.join(L, "*.log"))):
    if os.path.basename(f).startswith("ceil_") or f.endswith("fail.log"):
        continue
    print(subprocess.run(["python3", os.path.join(here, "260-turn106.py"), tok, f],
                         capture_output=True, text=True).stdout)
PY
}

# sourced (266-run.sh): the functions only
[ "${BASH_SOURCE[0]}" = "$0" ] || return 0
case "${1:-}" in
stage) stage ;;
run) shift; run "$@" ;;
cx) shift; cx "$@" ;;
sum) shift; sum "$@" ;;
*) sed -n '2,40p' "$0"; exit 1 ;;
esac
