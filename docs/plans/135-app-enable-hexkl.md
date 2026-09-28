# 135 — The Android app carries `ENABLE_HEXKL` whenever the prebuilt `libnntrainer.so` does

Issue: dlwlzzero/nntrainer#135 (p1; LEDGER rule 36, §3a "rebuild recipe until
#135 lands"; BENCHMARK #130 rows). Read against `htp_moe` @ `40970daf`.
Build-system only: no kernel, no IDL, no format tag, no CPU/NPU arithmetic
changes. The one thing it changes on the device is that `NNTR_HTP_FORWARD=1`
becomes reachable from a committed build; the switch stays off by default.

## 1. Goal and gate

Acceptance (issue body, Want 1–3, plus the supervisor's comment):

1. `Applications/CausalLM/build_android.sh --htp` produces a `libcausallm_core.so`
   with `ENABLE_HEXKL` defined. Measurable: `strings
   Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so | grep -c
   NNTR_HTP_FORWARD_KINDS` prints `1` (the string is in
   `lfm2_moe_causallm.cpp:89`, inside the `#ifdef ENABLE_HEXKL` block at `:60`;
   the committed app at `40970daf` prints `0` — #130 Notes ①). A build without
   `--htp` prints `0`. Windows / Tizen / Yocto: no file they read changes (§3).
2. A loud check: `build_android.sh` exits 1 when that count disagrees with
   `--htp`; the `hexagon-handoff` skill's sanity block carries the same line and
   the rule-36 sentence (a variant whose log lacks `[HTP] graph: init` is void).
3. The rebuild recipe in `hexagon-gates` rung 3 / `hexagon-handoff` no longer
   needs the local `-DENABLE_HEXKL=1` ndk-build line.
4. `libnntrainer.so` is unchanged by the PR (md5 before == after, or `.text`
   equal if ndk-build re-ran — 2f642ab7's method); the M1-GEMV / feed banner
   (`[HTP] moe m1 gemv: on … feed=vtcm source=default`) is untouched (it lives
   in `libnntrainer.so`, which does not change).

Standing gates: prefill ≥ −5 % of variant A and text identical to the CPU run
are not exercised — no device cell is needed (supervisor: "#136's sitting is
its first device use"). With `NNTR_HTP_FORWARD` unset the hooks return 0 at
`htp_compute_ops.cpp:1419-1421` before anything else, so A's decode path is
what #130's second set already measured (its A ran the `-DENABLE_HEXKL=1`
core `14008b7d…`, C ≡ A 6/6).

Host gate for this PR = rungs 0 and 3 (the build-system part of 3; the
device gtests are not touched). Rung 1 is unaffected: `jni/meson.build` is
read only under `platform == 'android'` (`meson.build:832-834`), and the host
`build/` never sees it. Rung 2 is not run (no file under `test/htp/` or
`htp_backend/` changes).

## 2. Where it lives

Verified at `40970daf`:

* `jni/meson.build:43-49` — `nntrainer_abi_defines` is `extra_defines` filtered
  to `-DENABLE_FP16=1` / `-DUSE__FP16=1`; `and_conf.set('NNTRAINER_ABI_DEFINES', …)`.
  `extra_defines` gains `-DENABLE_HEXKL=1` only inside `if get_option('enable-htp')`
  (`meson.build:296`, `:333`). **Change: one more `or` in the filter at `:45`, one
  comment line.**
* `jni/Android-prebuilt.mk.in:7` — `NNTRAINER_EXPORT_CFLAGS := @MESON_ARM_MARCH@
  @NNTRAINER_ABI_DEFINES@`, exported at `:15` / `:25` via `LOCAL_EXPORT_CFLAGS`
  of `ccapi-nntrainer` / `nntrainer`. Installed as
  `builddir/android_build_result/Android.mk` (`jni/meson.build:149`). No change.
* `Applications/CausalLM/jni/Android.mk:56-60` — includes the prebuilt mk; every
  module lists `nntrainer ccapi-nntrainer` in `LOCAL_SHARED_LIBRARIES`, so the
  exported cflags reach `causallm_core`, `causallm_api`, `nntrainer_causallm`,
  `test_api`, `nntr_quantize`, `nntr_quantize_stream`, `nntr_safetensors_info`,
  `unittest_causallm_models`. No change.
* `Applications/CausalLM/build_android.sh:273-284` — the `ndk-build` line and the
  `check_artifact` calls. **Change: the `strings` check after
  `check_artifact "libcausallm_core.so"` (`:283`).**
* `.claude/skills/hexagon-gates/SKILL.md:94-140` (rung 3). **Change: the
  `strings` line after the `readelf` line at `:98`, its pass condition, and one
  note on `--cache` after a pre-#135 `builddir`.**
* `.claude/skills/hexagon-handoff/SKILL.md:53-54` (template step 1). **Change: a
  three-line "sanity before pushing" block and the rule-36 void sentence.**

The ARM-side consumers of the define (no change, listed so the reviewer can
see what becomes live): `Applications/CausalLM/models/lfm2_moe/lfm2_moe_causallm.cpp:20`
(includes) and `:60-116` (`set_decode_graph_desc`, the `NNTR_HTP_FORWARD_KINDS`
mask); `Applications/CausalLM/layers/htp_decode_hook.h:25`, `:32-41` (the hook
calls `nntrainer::get_htp_ops()->decode_op_fp32`), `:92-94` (`decode_kv_seed_fp32`).
`get_htp_ops()` itself is declared only under `#ifdef ENABLE_HEXKL` in the
installed public header `nntrainer/tensor/cpu_backend/compute_ops.h:676-679`
(`set_decode_graph_desc` / `decode_op_fp32` at `:312` / `:328` are unconditional
virtuals, so the class layout does not depend on the define — this is an
API-visibility define, not a layout one, but it is the library's, not the app's).

Not touched (contract §2 consumers): the IDL and `generate_stub.sh`,
`HtpComputeOps`, `nntr_quantize_stream`'s format tag, the loader check, the
`NNTR_HTP_PROFILE` stage tables, `tools/htp_fc_report.py`. `test/jni/Android.mk`
does not include the prebuilt mk (`grep` finds only `Applications/CausalLM/jni/Android.mk`
and `jni/meson.build`), so the device gtests are unaffected.

## 3. Design

**Chosen: export the define from the prebuilt mk** — `jni/meson.build:45`
adds `-DENABLE_HEXKL=1` to `nntrainer_abi_defines`:

```meson
# ABI-affecting defines exported to prebuilt consumers via Android-prebuilt.mk:
# they gate the _FP16 type in the public headers (tensor_dim.h), and
# ENABLE_HEXKL gates get_htp_ops() (compute_ops.h) and the app's HTP decode
# hooks, so apps that link the prebuilts must compile with the same values.
foreach define : extra_defines
  if define == '-DENABLE_FP16=1' or define == '-DUSE__FP16=1' or define == '-DENABLE_HEXKL=1'
```

Why this one: the define describes how `libnntrainer.so` was configured, and the
app must match it to see the library's API — the same fact 2f642ab7 established
for the FP16 defines, through the channel it built for exactly that. It is off
for non-HTP builds by construction (`extra_defines` only carries it under
`enable-htp`, `meson.build:296/333`), and it flows to every ndk-build caller of
the app's `Android.mk` — `build_android.sh`, `build_api_lib.sh:104`,
`build_test_app.sh:125`, `run_unittest_android.sh:127` — without touching them.
Windows / Tizen / Yocto build the app inside the same meson invocation as the
library, where `extra_defines` already applies to both; they never read
`jni/meson.build` or the prebuilt mk.

**Rejected: an app-side flag** (`build_android.sh --htp` → `ndk-build
CAUSALLM_HTP=1` → `Android.mk` appends `-DENABLE_HEXKL=1` next to
`CAUSALLM_PROFILE`, `:50-52`). Two files instead of one, three other ndk-build
callers left behind, and it re-creates the desync the FP16 export removed: the
HTP option lives only in `builddir` (gates skill rung 3: "`--clean` … dropped
silently"), so `--cache` without `--htp` over an HTP builddir would again ship
an app without hooks against a library with them — #130's first pass, by
another route. The chosen mechanism makes that combination impossible; the
check below makes the remaining mismatch (a stale pre-#135 `builddir`) loud.

**The loud check** (`build_android.sh`, after `check_artifact "libcausallm_core.so"`):

```bash
# [#135] The HTP decode hooks (NNTR_HTP_FORWARD) are #ifdef ENABLE_HEXKL in
# the app and the define comes from the prebuilt Android.mk (jni/meson.build).
# A builddir configured before that export keeps its old Android.mk under
# --cache; this catches it, and catches the reverse (an HTP builddir reused
# without --htp). LEDGER rule 36.
n_htp=$(strings libs/arm64-v8a/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS)
if [ "$USE_HTP" -eq 1 ] && [ "$n_htp" -lt 1 ]; then
    log_error "libcausallm_core.so has no NNTR_HTP_FORWARD_KINDS: ENABLE_HEXKL missing (stale builddir/android_build_result/Android.mk? rebuild without --cache)"
    exit 1
elif [ "$USE_HTP" -eq 0 ] && [ "$n_htp" -ne 0 ]; then
    log_error "libcausallm_core.so carries ENABLE_HEXKL but --htp was not given: builddir is an HTP build (pass --htp, or rebuild without --cache)"
    exit 1
fi
log_info "ENABLE_HEXKL in libcausallm_core.so: $n_htp (expected $USE_HTP)"
```

`strings` is binutils' (`/usr/bin/strings` on the workstation); no new tool.
Contract §2 / doc 45 §3 (three walls, arena, `_det`, DMA behind compute): not
touched by this PR; nothing on the DSP changes. No `ponytail:` — there is no
corner cut.

## 4. Steps

1. **`jni/meson.build`** — the filter and comment above. Gate (rung 0): `git
   diff --stat` shows one file; nothing to clang-format.
2. **Regenerate the prebuilt mk and prove the export** — in a `builddir`
   configured with `--htp` (the main checkout's is: `intro-buildoptions`
   `enable-htp=true`, and its installed `Android.mk:7` today lacks the define —
   the cause, confirmed):
   ```
   (cd builddir && ninja install)     # meson regenerates on the meson.build change, reinstalls Android.mk
   grep NNTRAINER_EXPORT_CFLAGS builddir/android_build_result/Android.mk
   ```
   Pass: `NNTRAINER_EXPORT_CFLAGS := -march=armv8.2-a+fp16+dotprod+i8mm -DENABLE_FP16=1 -DUSE__FP16=1 -DENABLE_HEXKL=1`.
   Record `md5sum builddir/android_build_result/lib/arm64-v8a/libnntrainer.so`
   before and after; equal is the pass (if ndk-build re-ran and the md5 moved,
   compare `.text` as 2f642ab7 did: `llvm-objcopy -O binary --only-section=.text`
   on both, md5 equal).
3. **`build_android.sh`** — the check block. Gate (rung 3, the build-system
   part): `(cd Applications/CausalLM && ./build_android.sh --htp --cache)`.
   Pass lines in its output, in order: ndk-build's `[arm64-v8a] Compile++ :
   causallm_core <= lfm2_moe_causallm.cpp` (with `V=1` on the ndk-build line
   the compile command shows `-DENABLE_HEXKL=1` — optional, for the PR body),
   `[SUCCESS] Build completed successfully`, `[OK] libcausallm_core.so`,
   `[INFO] ENABLE_HEXKL in libcausallm_core.so: 1 (expected 1)`, then
   ```
   strings Applications/CausalLM/jni/libs/arm64-v8a/libcausallm_core.so | grep -c NNTR_HTP_FORWARD_KINDS   # 1
   strings Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so | grep -c 'graph: forward calls'  # 1
   readelf -d Applications/CausalLM/jni/obj/local/arm64-v8a/libnntrainer.so | grep -E 'libsdkl|libcdsprpc'  # both NEEDED
   ```
   Negative half of the gate, once: run the same `--htp --cache` with
   `builddir/android_build_result/Android.mk` temporarily reverted to the old
   `NNTRAINER_EXPORT_CFLAGS` line (`sed -i 's/ -DENABLE_HEXKL=1//'`), expect
   `[ERROR] … ENABLE_HEXKL missing …` and exit 1; restore with `ninja install`.
   The "stays off" half: the count-must-be-0 branch runs on every non-HTP
   build; a full non-HTP `build_android.sh` (≈ 10 min) is not required for
   this PR because the define enters `extra_defines` only under
   `enable-htp` (`meson.build:296/333`) — say so in the PR body rather than
   claim a run that was not made.
4. **Skills** — `hexagon-gates` rung 3: the `strings … NNTR_HTP_FORWARD_KINDS
   # 1` line after `readelf`, "1" added to the pass line, and one sentence: a
   `builddir` configured before #135 must run `(cd builddir && ninja install)`
   (or a build without `--cache`) once, since `--cache` skips meson and keeps
   the old installed `Android.mk`; the script's check names this case.
   `hexagon-handoff` template step 1 gains
   ```
   strings <libnntrainer.so> | grep -c 'graph: forward calls'        # 1
   strings <libcausallm_core.so> | grep -c NNTR_HTP_FORWARD_KINDS    # 1 (rule 36)
   ```
   and: "A variant whose log lacks its expected banner (`[HTP] graph: init …`
   for `NNTR_HTP_FORWARD=1`) is **void** — recorded as void, never read as
   'at A's speed' (rule 36)." Gate: rung 0 (docs only); the recipe no longer
   mentions the local `-DENABLE_HEXKL=1` ndk-build.
5. **PR into `htp_moe`**, `state:review`, body = the step-2 grep line, the
   step-3 pass lines and both md5s of `libnntrainer.so`. Commit subjects:
   `[build] Export ENABLE_HEXKL to the Android prebuilt consumers`,
   `[test] Fail build_android.sh when the app's ENABLE_HEXKL disagrees with --htp`
   (or one commit), `[docs] Gates and handoff skills: the app's ENABLE_HEXKL check`.

**Device measurement: none.** No step here needs the phone. The first device
use is #136's sitting, whose handoff (≤ 4 variants, A = the unchanged
reference built from the merged tree, full E2E, prompt 512, gen 64 / 512 /
1024) inherits the sanity block of step 4; its A is the first committed-tree
A that carries the hooks, and the #130 second-set A (`14008b7d…` core) is its
nearest prior reading.

## 5. Risks

* **Stale installed `Android.mk` under `--cache`** — the one host-vs-host gap
  this PR has. `--cache` skips meson entirely (`build_android.sh:206`), so a
  `builddir` from before the PR ships the old export line and the app still
  compiles the hooks out. The script's `exit 1` makes it visible on the
  workstation, and the handoff sanity line makes it visible in the artifact
  table before anything is pushed. Step 4's rung-3 note tells the builder the
  fix.
* **A's app binary changes md5** from every prior sitting's A: the hooks are now
  compiled in (returning 0 with the switch off). Handoff tables will show a new
  `libcausallm_core.so` md5 with an unchanged `libnntrainer.so`; the #130
  second set already ran this exact configuration (C ≡ A), so no A/A0 control
  is needed. The DMA-rate, DVFS and thermal gaps between sittings are not
  exposed by this PR (no device cell); they apply to #136's table as usual, and
  the table's per-row md5 + the sanity line separate "hooks absent" from
  "hooks slow", which #130's first pass could not.
* **Stale skel** — unaffected (IDL unchanged); a `0x8000040e` on #136's sitting
  is that sitting's skel, not this PR.
* **Address-space budget** — unchanged; nothing new is allocated on the DSP.
* **Other prebuilt consumers now see `get_htp_ops()`** — only code inside
  `#ifdef ENABLE_HEXKL` is affected, and the only such code in the app tree is
  the two files in §2; `nntr_quantize` / `nntr_quantize_stream` /
  `nntr_safetensors_info` compile the same sources they did (the define adds
  the hook calls to `causal_lm`-side layers they link, which return 0).

## 6. Docs to update

* `docs/htp_moe/LEDGER.md` rule 36: strike "(3) until #135 lands …" and point at
  the script check + skills; §3a "Rebuild recipe until #135 lands": replace with
  "since #135: `build_android.sh --htp` carries the define; a pre-#135 `builddir`
  needs one `ninja install` (or a build without `--cache`)". §3 ⑨: #135 done,
  next #136.
* `docs/htp_moe/BENCHMARK.md`: no measured row. The #130 artifact row's
  "uncommitted define (#135)" note gains "(fixed by #135, `<sha>`)"; Method:
  the next A is the first committed-tree A with the hooks compiled in.
* `docs/htp_moe/guide/01-run-it.md` (guide writer, after merge): the build line
  and the `strings` check.
