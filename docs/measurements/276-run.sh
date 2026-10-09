#!/usr/bin/env bash
##
# @file    276-run.sh
# @brief   #276 S1 sitting: the user's 2-bit (ternary, QS2CX_WH) Gemma-4 26B
#          expert file, E (one-PD decode) and A (#4415's hybrid) on the S1
#          build, S25 R3CY205ZMND
#
# Plan: docs/plans/276-2bit-experts-rev3.md (branch htp/276-plan) section 4
# S1, the minimal cell list of the S1 run (user, 2026-10-09). The cells, the
# cool start, the S1-ceiling column and the table come from 260-e-run.sh
# (sourced); only the model directory, the config directory and the base
# config change:
#   model   variant C, /data/local/tmp/nntrainer/gemma4_26b_ternary_fcqs4cx
#           (the user's files, never edited:
#           nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin, QS4CX FCs, Q4_0
#           embedding / untied lm_head, QS2CX_WH experts). Variant B (Q4_0
#           FCs, gemma4_26b_ternary): DM276= BIN276= FC276=Q4_0
#   lock    none (user, 2026-10-09: the sitting lock is retired)
#   configs /data/local/tmp/nntrainer/s276cfg/r2_<v>_p<p>_g<G>: the user's
#           nntr_config.json + the 260 r2 sitting's engine keys
#           (attn_proj / dense_ffn / attention htp), skip_prefill false,
#           attention_kv_dtype unset, lmhead_engine cpu for E and htp for A,
#           init_seq_len = 2 x prompt, max_seq_len >= init_seq_len
#
# usage:
#   276-run.sh sanity <log dir>  pulls the user's config / tokenizer into
#                                <log dir>/model, the device md5 of the file,
#                                a 4-tensor dd read and the palette / code /
#                                colsum check (276-qs2cx-check.py, host)
#   276-run.sh stage <log dir>/model/nntr_config.json   build + push configs
#   276-run.sh run <log dir>     cells 1-6
#   276-run.sh sum <log dir>     the table (260-e-run.sh sum)
set -u -o pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=260-e-run.sh
source "$HERE/260-e-run.sh"
DM=${DM276:-/data/local/tmp/nntrainer/gemma4_26b_ternary_fcqs4cx}
DC=${DC276:-/data/local/tmp/nntrainer/s276cfg}
B6=${B276:-/data/local/tmp/nntrainer/causallm/s276}
BIN=${BIN276:-nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin}
FC=${FC276:-QS4CX}

stage276() {
  local base=${1:?the user nntr_config.json, pulled from $DM} out d n
  out=$(mktemp -d)
  python3 - "$base" "$HERE" "$out" "$DM" "$BIN" "${P276:-512 1024}" "${G276:-64 512}" <<'PY'
import copy, json, os, sys
base_f, here, out, dm, binf, ps, gs = sys.argv[1:]
base = json.load(open(base_f))
base.update(skip_prefill=False, model_file_name=dm + "/" + binf,
            tokenizer_file=dm + "/tokenizer.json",
            attn_proj_engine="htp", dense_ffn_engine="htp",
            attention_engine="htp")
# keys the user's config omits and causal_lm.cpp reads unguarded
# (bad_word_ids: json type_error.302 on null); the 4-bit r2 base's values
for k, v in (("bad_word_ids", []), ("lora_alpha", 0), ("lora_rank", 0),
             ("lora_target", [])):
    base.setdefault(k, v)
base.pop("attention_kv_dtype", None)
if os.environ.get("KV276"):  # q8: the user's config of record (2026-10-09)
    base["attention_kv_dtype"] = os.environ["KV276"]
assert base["moe_layer_dtype"] == "QS2CX_WH", base["moe_layer_dtype"]
P = [int(x) for x in ps.split()]
prompts = {p: open(os.path.join(here, "260-prompt%d.txt" % p)).read() for p in P}
for v in ("E", "A"):
    for p in P:
        for g in [int(x) for x in gs.split()]:
            d = copy.deepcopy(base)
            d["sample_input"] = prompts[p]
            d["num_to_generate"] = g
            # INIT276=sum (#282): prompt + G; the 2 x prompt rule's q8 KV
            # buffers pushed the FC set out of the one-PD address space
            d["init_seq_len"] = p + g if os.environ.get("INIT276") == "sum" else 2 * p
            d["max_seq_len"] = max(base["max_seq_len"], d["init_seq_len"])
            assert p + g <= d["max_seq_len"]
            d["lmhead_engine"] = "cpu" if v == "E" else "htp"
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

sanity() { # the file's md5 on the device, and four expert tensors read back
  local L=${1:?log dir} t o s f
  mkdir -p "$L/model"
  for f in config.json nntr_config.json tokenizer.json; do
    $AD pull "$DM/$f" "$L/model/" >/dev/null
  done
  $AD shell "ls -l $DM/$BIN; md5sum $DM/$BIN" | tr -d '\r' | tee "$L/sanity_md5.txt"
  python3 "$HERE/276-qs2cx-check.py" layout "$L/model/config.json" "$FC" \
    layer0_expert0_gate_up layer0_expert0_down layer15_expert77_gate_up \
    layer29_expert127_down | tee "$L/sanity_layout.txt"
  tail -n +2 "$L/sanity_layout.txt" >"$L/sanity_tensors.txt"
  while read -r t o s _; do
    $AD exec-out "dd if=$DM/$BIN bs=65536 iflag=skip_bytes,count_bytes skip=$o count=$s 2>/dev/null" >"$L/$t.raw"
  done <"$L/sanity_tensors.txt"
  python3 "$HERE/276-qs2cx-check.py" check "$L/sanity_tensors.txt" "$L"
}

run276() {
  local L=${1:?log dir}
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  export COOL_QUICK=1
  echo "=== 276 S1 sitting $(date '+%F %T %Z') bin=$B6 uptime=$($AD shell cat /proc/uptime | tr -d '\r')"
  CLADDER=16 cell $B6 "$L" E 1024 64     # (1) load lines, prefill, text
  CLADDER=16 cell $B6 "$L" E 512 512     # (2) decode, misses, wait
  CLADDER=16 cell $B6 "$L" A 512 512     # (3) the same-file reference
  if CLADDER=32 cell $B6 "$L" E 512 512 c32; then # (4) plan L2
    CLADDER=32 cell $B6 "$L" E 1024 512 c32       # (5)
  else
    CLADDER=16 cell $B6 "$L" E 1024 512           # (5) at C = 16
  fi
  CLADDER=16 cell $B6 "$L" E 512 64 ppl NNTR_PPL=1 # (6)
  CLADDER=16 cell $B6 "$L" A 512 64 ppl NNTR_PPL=1
  echo "=== done $(date '+%F %T %Z')"
}

grid() { # the 2-bit best-version grid (#276, 2026-10-09): E at every prompt x G,
  # then A p1024 G512 as the one reference. Configs: DC276=<dir> P276="1024
  # 2048 4096" G276="64 512 1024" 276-run.sh stage <base>
  local L=${1:?log dir} p g
  mkdir -p "$L/done"
  exec > >(tee -a "$L/sweep.out") 2>&1
  export COOL_QUICK=1 CLADDER=16
  echo "=== 276 grid $(date '+%F %T %Z') bin=$B6 cfg=$DC"
  # TAG276=q8 names the cells <cell>_q8; SKIP4096=1 after a p4096 cell
  # took the phone down (fp16 KV at 8192 rebooted it three times)
  for p in 1024 2048 4096; do
    for g in 64 512 1024; do
      cell $B6 "$L" E "$p" "$g" "${TAG276:-}" ||
        { [ "$p" = 4096 ] && [ "${SKIP4096:-0}" = 1 ] && break; }
    done
  done
  cell $B6 "$L" A 1024 512 "${TAG276:-}"
  echo "=== done $(date '+%F %T %Z')"
}

case "${1:-}" in
stage) shift; stage276 "$@" ;;
sanity) shift; sanity "$@" ;;
run) shift; run276 "$@" ;;
grid) shift; grid "$@" ;;
sum) shift; TOK=${1:?log dir}/model/tokenizer.json; sum "$@" ;;
*) sed -n 2,28p "$0"; exit 1 ;;
esac
