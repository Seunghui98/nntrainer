# 250 — Repack the external int2 Gemma 4 .bin into QS2CX_WH

Issue: dlwlzzero/nntrainer#250 (p0). Base `htp_decode` @ `6904594ed`.
Spec = the issue body (decided 2026-10-07 on #229 S0); no planner run.

## 1. Goal and gate

A standing tool that turns the external converter's int2 Gemma 4 26B-A4B
`.bin` (the dummy now, the real checkpoint later, same script and
settings) into the `QS2CX_WH` file `htp_decode` loads.

Gate (issue):
* output size exact (dummy: 7,226,112,120 B in → 7,226,142,840 B out);
* `whUnpack2` of every repacked expert == the input's `sl/4` decode,
  element-wise, all 7,680 expert tensors; every other byte equal;
* the host loader opens the file with the 26B config;
* one `ponytail:` where the tool assumes the dummy's layout.

## 2. Where it lives

* `tools/htp/repack_int2_to_qs2cx.py` — python3, stdlib, streaming. Walks
  `writeGemma4Moe`'s order (`Applications/CausalLM/quantize_stream.cpp`)
  from `config.json` + the input's `nntr_config.json`; per expert permutes
  the codes inside each 256-byte tile (`sl/4` → `whCodeByte2` /
  `whCodeShift2`, `hvx_expand_i2i4.h`), inserts the palette `fe ff 00 01`,
  copies scale / colsum. Copies the `*.json` beside the input and writes
  `nntr_config.json` with `moe_layer_dtype: QS2CX_WH` and the new
  `model_file_name`.
* `tools/htp/repack_int2_check.cc` — built and run by the script after
  writing (`--verify-only` reruns it, `--skip-verify` skips). Reads both
  files by mmap with the script's segment table; per expert it decodes the
  input by the external rule, checks it against the input's own colsum
  (catches a different tile order or column placement on a new file; not
  a reordering of k inside a column), and compares it with the
  real `whUnpack2` (`htp_wh_palette.h`) of the output.

No change under `nntrainer/`, `Applications/` or `test/`.

## 3. Steps

1. Script + checker; synthetic two-layer file: repack, verify, and five
   single-byte mutations (expert code, copied byte, input colsum, palette,
   scale) each caught. Gate: `REPACK CHECK ... ok` / `FAIL`.
2. Run on the dummy into `/local/mnt/workspace/models/gemma4_26b/`.
   Gate: size line + `REPACK CHECK experts=7680 ... mismatches=0 ok`; md5.
3. Host load: `build_htp_host` `htp_e2e_test` (the inproc build that runs
   the gemma64 fixture) on the output, `--moe-engine htp`,
   `NNTR_MOE_CACHE_EXPERTS=16`. Gate: exit 0, `E2E gen` line, every expert
   registered. The host forward's numbers are not a gate (the dummy's FC
   Q4_0 is not host-readable, see the PR).
4. Rung 1 for a tools-only change: `ninja -C build`, clang-format-14 on the
   `.cc`. `run_host_checks.sh` not needed (nothing under `test/htp/`).
