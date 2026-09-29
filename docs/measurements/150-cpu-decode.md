# Measurement 150 (step 1b): where the ARM decode time goes between the MoE calls

Branch `htp/150-cpu-decode`, code @ `56ba0835` (the T set below was built
from that tree; later commits on the branch are docs only). Plan
`docs/plans/150-cpu-decode.md` §4 step 1b. Estimated device time:
**≈ 25 min**. Run by the orchestrator on the workstation (the phone stays
on USB).

## Why

1. On #141's Q path (dspqueue, now the default) a G=512 token is ≈ 26.5 ms,
   of which the 22 M==1 MoE calls are ≈ 22 × 0.714 = 15.7 ms; the other
   ≈ 10.8 ms is ARM work that has never been split on the TPS binary (the
   `--profile` build adds ≈ 20 ms/token, rule 1). This sitting reads that
   time per op kind with `NNTR_OP_TIME=1` at 8 / 6 / 4 threads, with the
   workers' context switches, the pinned core map and the cluster clocks.
   Step 2 picks at most two levers (plan §3.2 L1–L7) from this table and
   nothing else.
2. The timer must be inert (user's 2026-09-28 direction change): T8 must
   read within 1 % of A0's decode tok/s and give `bit_identical=1` MoE
   dumps and byte-identical `[PPL] decode` nll lines against A0. If the
   tok/s rule fails, the table is read as shares only.

This is a measurement-only step, so it runs prompt 512 at G=512 only (the
budget rows are G=512; plan §4 step 1b). G 64 / 512 / 1024 come with the
lever sitting (step 3).

### Variants (4; two binary sets, both skel `37468a7f…`)

| | set on the phone | env (beyond `NNTR_NUM_THREADS=8`) | expected banners |
|---|---|---|---|
| **A0** (reference, first) | `$D/a0/` = the #141b set (`07fb1938`, dspq default off) | `NNTR_HTP_DSPQ=1` | `[HTP] dspq: on …` once; no `[OP-TIME]` line |
| **T8** | `$D/` = this branch (`56ba0835`, dspq default on) | `NNTR_OP_TIME=1` | `dspq: on` once; `[OP-TIME] step` / `moe` / `node` lines at the end |
| **T6** | same | `NNTR_OP_TIME=1 NNTR_NUM_THREADS=6` | same |
| **T4** | same | `NNTR_OP_TIME=1 NNTR_NUM_THREADS=4` | same |

A0 is the plan's name: the #141b binary on the queue, i.e. `08de92b1`'s path
(the flip `c73384d2` changed only the default). It is **not** rule 40's
"A0 = `NNTR_HTP_DSPQ=0`" FastRPC fallback. Every log of every variant must
print `[HTP] dspq: on queue=0x… dsp_spin_us=1000 arm_spin_us=5000
buffers=2x65536 ion=y` once and `[HTP] dspq: close calls=N served=N bad=0
…` with N = 22 × G (rule 40). A log without the `on` line, or with
`dspq: off (…)`, is **void** (rule 36), never read as "at A0's speed".
`dspq: off (dspq_start … 0x8000040e)` means a stale skel (rule 3): stop.

Rule 15: no cell sets `NNTR_HTP_PROFILE` (level 2 swaps in the timed MoE
entry, level 3 inflates `ffn` ~5×). Rule 1: neither set is a `--profile`
binary (sanity below).

## Artifacts (`/local/mnt/workspace/htp_moe/150/set/`, `md5.txt` next to it)

| file | md5 | built with |
|---|---|---|
| `libnntr_hvx_skel.so` | `37468a7fdbf2e469589849598ca860ff` | #141b's (`07fb1938`, `test/htp/build.sh`, v79, HexKL 6.4.0.1). No DSP source or IDL changed since (`git diff 07fb1938 56ba0835` touches no file under `test/htp/` outside `host/`), so no rung 2 |
| `nntrainer_causallm` | `db0c4bc3129ef04ca9ec17d88d3092b7` | `build_android.sh --htp --cache` @ `56ba0835` (`jni/libs/arm64-v8a/`) |
| `libcausallm_core.so` | `44b60ecdc2a773048155864298aaea03` | same (`NNTR_HTP_FORWARD_KINDS` count 2; `OP-TIME` count 2) |
| `libnntrainer.so` | `178b6e126e3a2aac8d1d83ad2f658250` | same (`jni/obj/local/arm64-v8a/`, after `ninja -C builddir install`; NEEDED `libsdkl.so`, `libcdsprpc.so`; `dspq: on` count 1; `OP-TIME` count 1; `U dspqueue_` count 0) |
| `libccapi-nntrainer.so` | `e3f0a1243fbf28d8109e019e092288fc` | same (`jni/obj/local/arm64-v8a/`) |
| `a0/nntrainer_causallm` | `b1be9a062ae702af9e52b45510093826` | the #141b set, `07fb1938` (`141b/md5.txt`) |
| `a0/libcausallm_core.so` | `8204bf909354798c965b177ace694d15` | same |
| `a0/libnntrainer.so` | `de7e2b8c425d67e7f941117a5e719ec7` | same |
| `a0/libccapi-nntrainer.so` | `36fc2231eb7947ee93835309091670a8` | same |
| `libc++_shared.so` | `b1586b9b512712800fd36a24abac1c0a` | NDK r30 (from `141b/set/`) |
| `libsdkl.so` | `0ad4e22a70e4f135bce38ad8fd1e001b` | HexKL 6.4.0.1 (from `141b/set/`) |
| `prompt512.txt` (p01) | `fc65c1588dc66dd764c7013fe96cbb75` | `docs/measurements/77-prompt512.txt`, 512 tokens |
| `bitset-02-code.txt` … `bitset-08-short.txt` (p02–p08) | see `md5.txt` | staged for step 3; not run in this sitting |
| model `q40-qs4cx-wh`, `tokenizer.json` | `7b7867fab5…` / `7b8067a5…` | on the phone since #100 |

A0 runs from `$D/a0` with `LD_LIBRARY_PATH=.:..` (its four binaries first,
then the shared `libc++_shared.so` / `libsdkl.so`) and `ADSP_LIBRARY_PATH=..`
(the one skel). Not staged, never pushed: `builddir/.../libcdsprpc.so`
(rule 5).

Workstation sanity before pushing:

```
W=/local/mnt/workspace/htp_moe/150
(cd $W/set && LC_ALL=C md5sum -c ../md5.txt | grep -vc ': OK$')             # 0
strings $W/set/libnntrainer.so    | grep -c 'dspq: on'                        # 1
strings $W/set/a0/libnntrainer.so | grep -c 'dspq: on'                        # 1
strings $W/set/libnntrainer.so    | grep -c 'OP-TIME'                         # 1
strings $W/set/a0/libnntrainer.so | grep -c 'OP-TIME'                         # 0
strings $W/set/libcausallm_core.so $W/set/a0/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS   # 4 (2 each, rule 36)
strings $W/set/nntrainer_causallm $W/set/a0/nntrainer_causallm | grep -c 'per-layer-type totals'    # 0 (no profile binary)
find $W -name 'libcdsprpc*' | wc -l                                          # 0
```

Rebuild recipe if `$W` is not on the measuring workstation: `git checkout
56ba0835`, `source tools/htp/env.sh`, `export HEXKL_ROOT=…` explicitly,
`git submodule update --init --depth 1`, copy
`Applications/CausalLM/lib/libtokenizers_android_c.a` from another worktree,
`(cd Applications/CausalLM && ./build_android.sh --htp)` (on a fresh
`builddir`: `cd builddir && meson configure -Dprefix=$PWD/android_build_result
&& ninja install`, then `--htp --cache`); `libc++_shared.so` from the NDK
sysroot if missing. `--cache` does not rebuild `libnntrainer.so`: after any
source change run `ninja -C builddir && ninja -C builddir install` first.
The skel and the A0 set come from `/local/mnt/workspace/htp_moe/141b/set/`
(rebuild recipe in `141-dspq-moe.md`). A rebuilt set never matches the
table: record the md5s you push.

## Steps (workstation, phone on USB)

Shell setup once. Every `adb` names the serial (a second device may be on
USB). Clean run dir `s150`. Logs go to `logs1b/` (the `logs/` next to it
holds the earlier thread-count read on the #134 set; do not mix them).

```
cd /home/j2z0-lee/nntrainer-150 && git fetch -q && git checkout htp/150-cpu-decode && source tools/htp/env.sh
W=/local/mnt/workspace/htp_moe/150; L=$W/logs1b; mkdir -p $L $W/dump1b
S=R3CY10WM83Y      # adb devices: record the serial you use under Notes
D=/data/local/tmp/nntrainer/causallm/s150; M=/data/local/tmp/nntrainer/causallm/models/q40-qs4cx-wh; DD=/data/local/tmp/s150dump
therm() { adb -s $S shell dumpsys battery | grep -E 'level|temperature'; adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp; }
dir_of() { case $1 in A0*) echo "$D/a0";; *) echo "$D";; esac; }
env_of() { case $1 in A0*) echo "NNTR_HTP_DSPQ=1";; T8) echo "NNTR_OP_TIME=1";;
  T6) echo "NNTR_OP_TIME=1 NNTR_NUM_THREADS=6";; T4) echo "NNTR_OP_TIME=1 NNTR_NUM_THREADS=4";; esac; }
run() { # run <A0|A0p|T8|T6|T4> <G> <log name> [extra env]
  adb -s $S shell "cd $(dir_of $1) && \
    sed -i 's/\"num_to_generate\": [0-9]*/\"num_to_generate\": $2/' $M/nntr_config.json && \
    grep num_to_generate $M/nntr_config.json && md5sum $D/libnntr_hvx_skel.so nntrainer_causallm libnntrainer.so && \
    NNTR_NUM_THREADS=8 $(env_of $1) $4 LD_LIBRARY_PATH=.:.. ADSP_LIBRARY_PATH=$D \
    ./nntrainer_causallm $M \"\$(cat $D/prompt512.txt)\"" \
    2>&1 | tee $L/$3.log | grep -E '^(prefill|generation|total|peak memory)|dspq:|moe m1 gemv|pinning|\[OP-TIME\] (step|moe)|libnntr_hvx_skel|nntrainer_causallm$|libnntrainer.so$'
}
# The generated text of a log: everything before the summary, minus the
# banners and any [OP-TIME] line stderr put inside the streamed text.
strip() { sed -n '/^=====/q;p' "$1" | perl -0pe 's/\[HTP\] dspq: [^\n]*\n//g; s/\[OP-TIME\][^\n]*\n//g' |
  grep -v 'moe m1 gemv\|libnntr_hvx_skel\|nntrainer_causallm\|libnntrainer.so\|num_to_generate'; }
```

### 0. Device state (1 min)

```
adb devices                           # note every serial listed
therm | tee -a $L/therm.log           # checkpoint 0 (battery %, °C·10, zone0 m°C)
```

Screen off, charger in, cool start.

### 1. Install (≈ 3 min) — model reused

```
adb -s $S shell ls -l $M/nntr_lfm2_8b_a1b_q40_arm.bin                        # 4316133120
adb -s $S shell "rm -rf $D $DD && mkdir -p $D $DD"
adb -s $S push $W/set/. $D/
adb -s $S shell "chmod 755 $D/nntrainer_causallm $D/a0/nntrainer_causallm"
adb -s $S shell "cd $D && md5sum \$(find . -type f | sed 's|^\./||' | sort)" | tee $L/md5_device.log
diff <(sort -k2 $W/md5.txt | tr -d '\r') <(sort -k2 $L/md5_device.log | tr -d '\r') && echo MD5 OK
```

Config (as #141: greedy, `init_seq_len: 512`, `moe_engine: htp`,
`moe_htp_layers: ""`), re-applied unconditionally:

```
adb -s $S shell "cd $M && \
  sed -i 's/\"do_sample\": true/\"do_sample\": false/' generation_config.json && \
  sed -i 's/\"bad_word_ids\": \[\]/\"bad_word_ids\": [124900]/' nntr_config.json && \
  (grep -q moe_engine nntr_config.json || sed -i 's/\"bad_word_ids\": \[124900\],/\"bad_word_ids\": [124900],\n    \"moe_engine\": \"htp\",/' nntr_config.json) && \
  grep -H do_sample generation_config.json && grep -H -E 'bad_word_ids|num_to_generate|init_seq_len|_engine|_htp_layers' nntr_config.json"
```

Expected echo: `do_sample": false`, `bad_word_ids": [124900]`,
`init_seq_len": 512`, `moe_engine": "htp"`, `moe_htp_layers": ""`, no other
`_engine` key.

### 2. (a) Sanity at G=8 (≈ 2 min)

```
for v in A0 T8 T6 T4; do run $v 8 sanity_$v; done
grep -c 'dspq: on' $L/sanity_*.log                                          # 1 each
grep -c 'OP-TIME' $L/sanity_A0.log                                          # 0
python3 tools/htp/op_time_report.py $L/sanity_T8.log | head -3              # a table header, tokens=8
grep -h 'pinning' $L/sanity_*.log                                           # nothing
```

Expected: every log `dspq: on` once and `dspq: close calls=176 served=176
bad=0`; the T logs `[OP-TIME] step tokens=8 …` and `[OP-TIME] moe
calls=176 …`; the md5sum lines show the A0 binaries (`b1be9a06…`,
`de7e2b8c…`) for A0 and the T binaries (`db0c4bc3…`, `178b6e12…`) for T*.
A `pinning thread on cpu<n> failed!` line is a finding, not a stop: record
it (plan §5 affinity risk).

### 3. (b) tok/s + table, prompt 512, G=512, mirrored (≈ 6 min)

```
for v in A0 T8 T6 T4; do run $v 512 ${v}_G512_r1; done
for v in T4 T6 T8 A0; do run $v 512 ${v}_G512_r2; done
therm | tee -a $L/therm.log                                                 # checkpoint 1
```

Expected in every log: `prefill: 512 tokens, … TPS`, `generation: 512
tokens, … TPS`, `generation(last 64): 64 tokens, … TPS`, `peak memory`, one
`moe m1 gemv` banner, `dspq: on` once, `dspq: close calls=11264
served=11264 bad=0`. T logs: `[OP-TIME] step tokens=512 …`, `[OP-TIME] moe
calls=11264 …` and ≈ 250 `[OP-TIME] node` lines. T6 / T4 prefill is lower
by design (#150's first read: −8 / −17 %); only decode is read here.

### 4. (c) Task, clock and core-map snapshot during one extra T8 run (≈ 1 min)

```
snap() { adb -s $S shell 'p=$(pidof nntrainer_causallm); echo "pid=$p uptime=$(cut -d" " -f1 /proc/uptime)";
  for t in /proc/$p/task/*; do printf "tid=%s cpu=%s " ${t##*/} $(sed "s/.*) //" $t/stat | cut -d" " -f37);
    grep -E "^(Name|Cpus_allowed_list|voluntary_ctxt_switches|nonvoluntary_ctxt_switches):" $t/status | tr "\t\n" "  "; echo; done'; }
freq() { adb -s $S shell 'for i in $(seq 25); do printf "uptime=%s " $(cut -d" " -f1 /proc/uptime);
  for p in /sys/devices/system/cpu/cpufreq/policy*; do printf "%s=%s " ${p##*/} $(cat $p/scaling_cur_freq); done; echo; sleep 0.2; done'; }
adb -s $S shell 'for p in /sys/devices/system/cpu/cpufreq/policy*; do echo "${p##*/} cpus=$(cat $p/related_cpus | tr " " ,) max=$(cat $p/cpuinfo_max_freq)"; done;
  for c in /sys/devices/system/cpu/cpu[0-9]; do echo "${c##*/} cluster=$(cat $c/topology/cluster_cpus_list)"; done' | tee $L/topology.log
run T8 512 snap_T8 > /dev/null &
RP=$!
adb -s $S shell 'until pidof nntrainer_causallm > /dev/null; do sleep 0.1; done'
sleep 20; snap | tee $L/snap_20.log | head -3
freq > $L/freq.log
snap | tee $L/snap_25.log | head -3                          # ≈ 25 s: freq takes ≈ 5 s
adb -s $S shell cat /sys/class/thermal/thermal_zone0/temp | tee -a $L/snap_zone0.log
wait $RP; grep -E '^generation:|dspq: close' $L/snap_T8.log
```

A G=512 run takes ≈ 32 s from launch (≈ 17 s model load, then prefill and
≈ 13 s of decode, #141b logs), so 20 s and ≈ 25 s are both inside the
decode. Check: `snap_25.log` must still list a pid and ≈ 12 tids (main, 7
workers, the FastRPC / dspq threads). If the first snapshot shows no pid
or the run already printed `generation:` before 25 s, record it and use
`sleep 18` instead. Workers' `voluntary_ctxt_switches` difference between
the two snapshots, divided by the tokens decoded in between
(≈ 5 s × 38 tok/s), is L3's trigger (≈ 22 × 7 per token means the
workers sleep through every MoE call).

Optional, no rebuild (1 min, not a tok/s cell):
`adb -s $S shell "cd $D && NNTR_NUM_THREADS=8 NNTR_OP_TIME=1 LD_LIBRARY_PATH=.:.. ADSP_LIBRARY_PATH=$D simpleperf stat --per-thread -e task-clock,context-switches,cpu-migrations ./nntrainer_causallm $M \"\$(cat $D/prompt512.txt)\"" > $L/simpleperf_T8.log 2>&1`
(G stays 512 from step 3 / 4).

### 5. (d) Dumps + decode nll lines, prompt 512, G=64 (≈ 5 min, ≈ 0.8 GB per variant on the phone)

```
for v in A0 T8 A0p; do
  adb -s $S shell "rm -rf $DD/$v $DD/ppl_$v.ids && mkdir -p $DD/$v"
  run $v 64 dump_$v "NNTR_HTP_DUMP=$DD/$v NNTR_PPL_DECODE=$DD/ppl_$v.ids"
  rm -rf $W/dump1b/$v && adb -s $S pull $DD/$v $W/dump1b/$v > /dev/null && adb -s $S shell "rm -rf $DD/$v"
done
therm | tee -a $L/therm.log                                                 # checkpoint 2
E="python3 tools/htp/htp_dump_eval.py"
$E --label a0p $W/dump1b/A0 $W/dump1b/A0p | tail -1     # null check: must be bit_identical=1
$E --label t8  $W/dump1b/A0 $W/dump1b/T8  | tail -1     # step-1 gate: bit_identical=1
nll() { grep -o '\[PPL\] decode step=.*' "$1"; }
cmp <(nll $L/dump_A0.log) <(nll $L/dump_A0p.log) && echo "nll A0p == A0"
cmp <(nll $L/dump_A0.log) <(nll $L/dump_T8.log)  && echo "nll T8 == A0"
nll $L/dump_A0.log | wc -l                                                   # 64: one per decode step (dspq close calls=1408 = 22 × 64)
grep -h '\[PPL\] decode tokens=' $L/dump_A0.log $L/dump_T8.log $L/dump_A0p.log
```

`a0p` not `=1` means the sitting cannot judge the inertness gate (A0 is
not bit-stable run to run): record it and go on. `t8` `bit_identical=0`
or a differing nll line is a defect of the timer: record the `first_diff`
and the first differing line, and stop the fold (the table is void).

### 6. Report and checks on the workstation (1 min)

```
R="python3 tools/htp/op_time_report.py"
for r in 1 2; do $R $L/T8_G512_r$r.log; $R $L/T6_G512_r$r.log --base $L/T8_G512_r$r.log; $R $L/T4_G512_r$r.log --base $L/T8_G512_r$r.log; done | tee $L/report.txt
grep -H '^generation:' $L/A0_G512_r?.log $L/T?_G512_r?.log
grep -c 'dspq: on' $L/*_G512_r?.log $L/dump_*.log $L/snap_T8.log               # 1 each (rule 40)
grep -l 'dspq: off\|HTP-PROFILE' $L/*.log                                     # nothing
grep -h 'dspq: close' $L/*_G512_r?.log | awk '{print $4, $5, $6}' | sort | uniq -c   # calls=11264 served=11264 bad=0
grep -c 'OP-TIME' $L/A0_G512_r?.log $L/dump_A0*.log                           # 0 each
grep -h 'pinning' $L/*.log                                                    # nothing, else record it
for v in T8 T6 T4; do for r in 1 2; do printf '%s r%s text vs A0: ' $v $r
  cmp -s <(strip $L/A0_G512_r$r.log) <(strip $L/${v}_G512_r$r.log) && echo identical || echo DIFFERENT; done; done
cat $L/therm.log
```

Each report must print a table whose `total` equals the log's token time
and exit 0 (it exits 1 on a node / step token-count mismatch or a timed
sum above the token time by > 2 %). Text differing from A0 in any tok/s
cell is a defect (the thread count does not change the per-element
arithmetic: #150's first read, G=8 dumps 6 vs 8 and 4 vs 8
`bit_identical=1`).

Then fill the tables below, commit this file on the branch, push, and set
#150 to `state:measured`.

## Gates (plan 150-cpu-decode.md §1, step 1)

| # | check | pass |
|---|---|---|
| S1 | T8 not inflated: T8 decode tok/s (all), G=512, mirrored mean, vs A0's | within 1 % of A0; otherwise the table is read as shares only |
| S2 | T8 inert: `htp_dump_eval` T8 vs A0 (step 5) | `bit_identical=1` on every file; the null A0p vs A0 must be `=1` |
| S3 | nll lines (step 5) | `[PPL] decode step=… nll=%.17g` byte-identical, T8 vs A0 and A0p vs A0 |
| S4 | banners (rule 40) | `dspq: on` once in every log; `close calls=N served=N bad=0`, N = 22 × G; no `dspq: off`; no `OP-TIME` in any A0 log |
| S5 | text in every tok/s cell (step 6) | T8 / T6 / T4 ≡ A0 of the same run |

## Results

Run 2026-09-29 10:50–11:05 KST by the orchestrator, unit `R3CY10WM83Y` (a
second device `R3CN80CW3FY` on USB, never addressed). Device md5 = md5.txt
(MD5 OK). Thermal (battery °C·10 / zone0 m°C): 311 / 41300 before the G=512
cells, 336 / 61000 after. Every log: `dspq: on` once, `dspq: close
calls=N served=N bad=0`; no `pinning` line.

### tok/s, prompt 512, G=512 (mirrored A0 T8 T6 T4 | T4 T6 T8 A0)

| variant | prefill r1 / r2 | decode r1 / r2 | decode mean |
|---|---|---|---|
| A0 (#141b set, dspq on) | 539.5 / 455.9 | 38.19 / 36.78 | 37.48 |
| T8 (timer on, 8 threads) | 503.9 / 422.8 | 38.14 / 38.80 | 38.47 |
| T6 | 412.6 / 380.4 | 39.27 / 38.67 | 38.97 |
| T4 | 357.3 / 344.8 | 38.56 / 36.72 | 37.64 |

Timer inertness: T8 is within A0's own r1/r2 spread (A0 spread 1.40 tok/s);
**MoE dumps A0 vs T8 `bit_identical=1` (2862 files), null A0 vs A0' `=1`,
the 64 decode nll lines of T8 and A0' equal A0's, texts T8 = A0 in both
runs.** Thread count moves decode by at most ≈ 1–4 % inside the spread and
costs prefill 8–35 %, as #150's first read said.

### Per-op decode table (T8 run 1; run 2 agrees within 3 % on every row)

```
[OP-TIME] /local/mnt/workspace/htp_moe/150/logs1b/T8_G512_r1.log: tokens=512 token=26.220 ms (38.14 tok/s)
kind                        ms/token   share  MB/token    GB/s  avg/min
FC conv in_proj                2.491    9.5%     127.4    51.1     1.12
FC conv out_proj               0.800    3.1%      42.5    53.1     1.10
FC attn qkv (+q/k norm)        0.568    2.2%      21.2    37.4     1.15
FC attn o                      0.264    1.0%      14.2    53.6     1.09
FC dense FFN (+swiglu)         0.867    3.3%      49.5    57.2     1.12
conv block (fused)             0.000    0.0%       0.0       -        -
attention (mha_core)           1.473    5.6%       0.0       -     1.81
conv1d + gate                  0.146    0.6%       0.4     3.0     1.34
RMSNorm                        0.114    0.4%       0.4     3.5     1.34
residual add                   0.104    0.4%       0.0       -     1.32
MoE CPU part                   0.412    1.6%       5.8    14.0        -
MoE wait                      16.077   61.3%       0.0       -        -
lm_head                        2.401    9.2%     147.5    61.4     1.05
embedding                      0.003    0.0%       0.0       -     1.33
sampling                       0.107    0.4%       0.0       -        -
register                       0.077    0.3%       0.0       -        -
other                          0.036    0.1%       0.0       -     1.83
unattributed                   0.280    1.1%       0.0       -        -
total                         26.220  100.0%
```

**Reading.**
* **The MoE wait is 16.0 ms of 26.2 (61 %)** — 484 MB of expert weights
  at ≈ 30 GB/s in-app (rule 41). Everything the CPU does is the other 10 ms.
* **The CPU already streams its weights at 51–63 GB/s**: conv in_proj
  51–53, out_proj 53–54, attn o 54–55, dense FFN 57–60, lm_head 61–63 GB/s
  (of an 85.3 GB/s LPDDR5X spec peak). FC + lm_head = 7.4 ms for 402 MB.
  Only the fused qkv (+ q/k norm) is lower (37–41 GB/s, 0.56 ms).
* Attention on the CPU is 1.43–1.47 ms/token at pos 512–1023 (fp16 KV,
  avg/min 1.6–1.8: the cost grows with position).
* Norms, adds, conv1d, sampling, register, embedding together ≈ 0.6 ms;
  unattributed 0.28–0.29 ms.
* During decode the CPU clocks sit at 2.0–2.7 GHz of 3.53 (cpus 0–5) and
  2.0–3.1 of 4.47 GHz (cpus 6–7) (`freq.log`, 25 samples).

**Consequence for #150's levers:** the CPU side is close to its byte floor
(≈ 402 MB at 60 GB/s ≈ 6.7 ms vs 7.4 measured for FC + lm_head); the
bit-preserving CPU headroom is ≈ 0.5–1 ms (the qkv row, attention,
unattributed), not the 1.5–2.5 ms the plan estimated. The big term is the
DSP's MoE read at ≈ 30 GB/s while the CPU reads at 55–63 GB/s.
 (fill in)

Unit, date, KST window: …; device `md5sum` = `md5.txt`? …; thermal
checkpoints (battery °C·10 / zone0 m°C): 0 …, 1 …, 2 ….

### tok/s (prompt 512, G=512; mirrored A0 T8 T6 T4 | T4 T6 T8 A0)

| variant | prefill r1 / r2 | decode (all) r1 / r2 | decode mean | vs A0 | last 64 r1 / r2 | peak RSS | text ≡ A0 | skel md5 (device) |
|---|---|---|---|---|---|---|---|---|
| A0 | | | | — | | | (reference) | |
| T8 | | | | | | | | |
| T6 | | | | | | | | |
| T4 | | | | | | | | |

Reference (#141 sitting, same unit, Q = today's default): decode 38.55 /
36.99 (mean 37.77) tok/s at G=512, prefill 483.5 / 436.5; G=64 39.40.
Goal ≥ 50 decode, prefill ≥ −5 % of A. Expected here: T8 ≈ A0 (the timer
costs ≈ 250 nodes × 2 clock reads ≈ 15 µs/token by estimate, ≈ 0.06 %);
T6 / T4 decode within ≈ ±2 % (#150's first read on FastRPC: +1.8 / +1.5 %).

### Per-kind table, G=512 (paste `report.txt`; r1 and r2)

Expected order of magnitude (plan §1, 408 MB/token of weights outside the
MoE): token ≈ 26.5 ms; MoE wait ≈ 15.7; lm_head 147.5 MB, conv in_proj
127.4, conv out_proj 42.5, dense FFN 49.5, attention FCs 35.4 MB/token;
at 50 GB/s the FC kinds and lm_head together ≈ 8.2 ms. The GB/s column
settles whether the FCs run at ≈ 38, 50 or near 67.9 GB/s.

```
(paste $L/report.txt)
```

| kind | T8 ms/token r1 / r2 | T8 GB/s | T8 avg/min | T6 − T8 | T4 − T8 |
|---|---|---|---|---|---|
| FC conv in_proj | | | | | |
| FC conv out_proj | | | | | |
| FC attn qkv (+q/k norm) | | | | | |
| FC attn o | | | | | |
| FC dense FFN (+swiglu) | | | | | |
| conv block (fused; 0 unless a conv_block engine is set) | | | | | |
| attention (mha_core) | | | | | |
| conv1d + gate | | | | | |
| RMSNorm | | | | | |
| residual add | | | | | |
| MoE CPU part | | | | | |
| MoE wait | | | | | |
| lm_head | | | | | |
| embedding / sampling / register | | | | | |
| other | | | | | |
| unattributed | | | | | |
| total | | | | | |

`[OP-TIME] moe` lines (T8 r1 / r2): `calls=… cpu_us=… call_us=…` → call
µs/call = … (A0's #141 level-2 M==1 `host` = 700.0 + 14.1).

### Snapshot (c)

| thread (Name) | tid | Cpus_allowed_list | cpu (stat 39) @20 s / @25 s | voluntary Δ | nonvoluntary Δ | Δ per token |
|---|---|---|---|---|---|---|
| main | | | | | | |
| worker 1..7 | | | | | | |
| FastRPC / dspq threads | | | | | | |

Tokens decoded between the snapshots: … (uptime Δ × T8 decode tok/s).
`topology.log`: policies and clusters …; `freq.log`: per-policy
`scaling_cur_freq` min / median / max vs `cpuinfo_max_freq` …; zone0 ….
Pinning failures: ….

### Gates

| # | result | pass |
|---|---|---|
| S1 | | |
| S2 | | |
| S3 | | |
| S4 | | |
| S5 | | |

## Notes from the run

Thermal, first-run page faults, FARF / AEE errors, a second device on USB,
anything stale.
