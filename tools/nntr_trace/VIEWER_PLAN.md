# nntr_trace viewer: implementation plan (W0-W13)

Audience: the agent implementing this. Everything below is specific enough
to start from without re-deriving the design; where a fact must be checked
on the real data (QNN optrace field names), the item says so.

## 0. Ground rules

- **Files.** `tools/nntr_trace/viewer.html` (single file, no build step, no
  external scripts: the artifact CSP blocks everything except cdnjs, and the
  local file must work from `file://`), `make_sample_trace.py`,
  `stage_us_to_trace.py`, a new `summarize.py`, a new `bundle.py`, a new
  `test/` directory. Python is **stdlib only** (runs on whatever machine the
  phone is plugged into).
- **Viewer structure today** (`viewer.html`, ~400 lines of JS in one IIFE):
  `parse(doc)` builds the model `T = {procs, rows, height, tmin, tmax,
  flows, counters, md, phases, all}` and assigns `_d` (depth), `_cat`, `_th`,
  `_op`, `_label` on every X event; `union / inter / minus / spans` are the
  interval primitives; `metrics(range)` and `kernels(range)` derive the
  numbers; `draw()` paints the canvas (ruler, rows, slices, flows, gutter);
  `renderPane()` fills the bottom panel by `tab`; `hit()`, `zoom()`, `pan()`,
  `fit()`, `load()` do interaction. `colorBy` is `op | name | engine`.
  Categories are the `cat` strings in the sample generator (`host.cpu`,
  `host.wait`, `host.rpc`, `host.layer`, `host.phase`, `host.load`,
  `dsp.entry`, `dsp.call`, `dsp.hmx`, `dsp.hvx`, `dsp.dma`, `dsp.sync`).
- **Sample data** is synthetic (`make_sample_trace.py`); when a feature needs
  a field the sample does not emit, add it to the generator in the same
  change and say so in the commit. Never invent a field the real tracer
  (docs/backend_guide/HTP_TRACE_PROFILER.md §4) would not have; extend §4 if
  a new `args` key is introduced.
- **Testing.** `test/check.js` runs headless Chromium
  (`executablePath: '/opt/pw-browsers/chromium'`, `NODE_PATH=$(npm root -g)`
  for playwright, as used during P0) against a bundled page and asserts:
  zero `pageerror`, and the numbers exposed on `window.__nntr` (see W0).
  `test/test_summarize.py` (unittest) checks `summarize.py` against the
  sample and against the viewer's JSON (fixture written by `check.js`).
  Run both before every commit; there is no CI for this directory, so the
  commit message states the command and its result.
- **Commits.** Per AGENTS.md: `git commit -s`, `[tools] <topic>` prefix,
  `Co-authored-by:` trailer for agent-authored work, one topic per commit.
  One PR per batch (§2). Keep `README.md` in this directory current.
- **Design constraints to keep.** chrome://tracing look (17 px rows,
  catapult palette, black slice text, gray chrome, bottom Analysis panel);
  light and dark themes via the existing tokens; phone width must not scroll
  horizontally except inside the timeline and tables.

## 1. Work items

Each item: goal, model/data changes, UI, algorithm, acceptance. Sizes are
rough (S < 100 lines, M < 300, L > 300).

### W0. Foundation refactor (M) - do first

- Split `parse(doc)` into `parseEvents(doc) -> model` (events, threads,
  counters, flows, metadata, depths) and `layoutRows(models, opts) -> rows`
  (row geometry; `opts.collapsed` is a Set of `pid` / `pid:tid` keys). `T`
  becomes `{models: [...], rows, ...}` so W3/W9 can show more than one trace;
  a single model stays the default path.
- Register analysis tabs in a table: `TABS = [{id, label, render(pane, ctx)}]`,
  rendered from that list; new tabs are additive.
- Expose `window.__nntr = { model: () => T, metrics, kernels, metricsJSON,
  setRange, select, view }` for tests and for W12.
- Add `bundle.py`: `bundle.py viewer.html -o out.html --trace name=path ...`
  embeds traces into `<script id="embedded-traces">` (escaping `</`). This
  replaces the ad-hoc scratch script used in P0. `README.md` documents it.
- Add `test/check.js` + `test/run.sh` (generates samples, bundles, runs
  check) and a `test/fixtures/` directory (git-ignored outputs).
- Acceptance: page renders the three samples with no errors; `__nntr.metrics`
  on the as-built sample returns `compression` 1.22 +- 0.01 and on the
  pipelined sample 1.62 +- 0.01 (values from P0).

### W1. Units-busy strip (S)

- Goal: parallel compression over time. One synthesized counter row per DSP
  process named `units busy`, value = number of lanes (HMX, each HVX_i, DMA)
  with a slice covering `t`.
- Algorithm: sweep line over all `dsp.hmx | dsp.hvx | dsp.dma` events
  (+1 at ts, -1 at ts+dur), emit `[t, count]` on every change; coalesce
  runs shorter than 1 px at the current zoom in `draw()` (draw max of the
  run). Also `host threads busy` for the host process from `host.cpu`.
- UI: rendered like the existing counter rows but as a heat strip (0 =
  surface, max = the darkest palette step of one hue; a 6-step ordinal ramp
  of blue is fine); hover shows `n / max`. Sits directly under the process
  header so it is read before the lanes.
- Acceptance: on the as-built sample the strip never exceeds 2 inside FC
  calls (HMX then HVX serialized, DMA prefetch overlapping); on the
  pipelined sample it reaches 3.

### W2. Drag range selection + `m` (M)

- Goal: any time range as the analysis range, like chrome://tracing.
- Interaction: drag in the ruler area (`y < HDR`) or shift+drag anywhere
  creates a selection `{t0, t1}` drawn as the existing `--ovl` overlay with
  two 1 px edge lines and the duration printed in the ruler. `m` marks the
  selected slice's span as the range. `Esc` clears. The range select gets
  a `Range: selection` option that is chosen automatically.
- Model: `range` already exists; add `range.kind = 'phase' | 'selection'`.
  Selection persists across zoom; the ruler overlay is clipped to the view.
- Acceptance: selecting exactly one `layer0` span (via `m` on the layer
  slice) gives the same numbers as the Layers tab row for `layer0`.

### W3. A/B compare (L)

- Goal: two traces, one screen; op-level deltas.
- Loading: `Compare...` button (file input, or a second embedded sample via
  a select). The second trace becomes `T.models[1]`; `layoutRows` stacks its
  processes under the first with headers prefixed `A:` / `B:` and both
  aligned at their own `tmin`. A `View: A | B | A+B` select hides the other
  model's rows.
- Compare tab (new `TABS` entry, enabled only with two models):
  1. Wall-time buckets side by side (two stacked bars, same scale) and
     `compression`, `idle`, `transport` as A / B / delta.
  2. Kernels joined on `engine|label`: calls, total, mean, cy/elem for A and
     B, delta % on total and on cy/elem, sorted by |delta total|; rows only
     in one side are listed with `-`.
  3. Layers joined on name (+phase): wall A, wall B, delta.
  Range applies per model in *relative* time (same phase index if both
  have phases, else whole trace).
- Acceptance: as-built vs pipelined shows `HMX ∥ HVX` 0 -> >0 and
  compression 1.22 -> 1.62; `qkv_proj: dequant i32->f32` mean delta 0 (same
  work, moved in time).

### W4. Transport model scatter (M)

- Goal: fixed + per-byte FastRPC cost, and the calls that break it.
- Data: pair `host.rpc` events on the seam thread in order: a `marshal *`
  followed by the next `return *` with the same suffix is one call;
  `bytes = marshal.args.bytes + return.args.bytes`, `us = sum(dur)`. The
  matching `host.wait` slice is the one on `main` that contains both.
- Fit: ordinary least squares `us = a + b * bytes` over the calls in range;
  report `a` (fixed cost, µs) and `1/b` as MB/s. Residual = us - fit;
  outliers = |residual| > 2 sigma.
- UI: new `Transport` tab: inline SVG scatter (x bytes log-scaled if the
  span exceeds 100x, y µs), the fit line, outliers marked and listed in a
  table (call, bytes, us, residual) with click -> select the wait slice.
  Text line: `n calls, fixed a µs, b µs/MB (≈ X MB/s)`; compare with the
  doc-34 numbers (963 µs @ M=1024, 326 µs @ M=1) in a hint.
- Sample: the generator already emits `bytes` on both seam events; add a
  small random jitter (deterministic seed) to transport so the fit has
  residuals.
- Acceptance: on the sample the fit recovers `a` within 15 % of 326 µs.

### W5. Pool tail table (S)

- Goal: automate the `tiles >= threads x 4` audit (doc 36 T6).
- Data: every `pool_run *` slice (`dsp.sync` on the DSP main thread) with
  `args.pool` (new: unique id per run, emitted by the generator) and the
  `dsp.hvx` unit slices carrying the same `args.pool`. Per run: `units`,
  `max_unit`, `mean_unit`, `tail = max/mean`, `join_gap = span - max_unit`.
- UI: section at the bottom of the Engines tab: table grouped by pool kind
  (the name after `pool_run `): runs, mean tail, worst tail, units vs
  `hvx_threads * 4` rule (flag when `units < threads * 4`), flag tail > 1.3
  with the `serious` pill. Row click selects the worst run.
- Acceptance: the sample's `quant f32->u8 AH` (units = K/32) is flagged for
  K < 768 shapes only.

### W6. HMX utilization (S)

- Goal: separate "slow kernel" from "lots of work".
- Data: for slices with `args.M, K, N` (`dsp.hmx`, `dsp.call`, `host.wait`):
  `ops = 2 * M * K * N * (args.n_handles || 1)`, `TOPS = ops / dur_us / 1e6`.
  Peak comes from `metadata.hmx_peak_tops` (nullable; the generator writes
  `null` with a comment: measure on device, do not guess).
- UI: Selection tab rows `MACs`, `TOPS`, `% of HMX peak` (only when peak
  known). Kernels tab gains a `TOPS` column for HMX rows.
- Acceptance: the sample's decode `micro-mm (M=2, 62 rows pad)` shows TOPS
  two orders below the prefill chunk rows, which is the padding tax the
  docs describe.

### W7. Warnings banner + fallback marking (M)

- Goal: never read a flattering trace. Surface `metadata.dropped
  {host, dsp}`, `metadata.clock_sync.violations`, `metadata.htp_enabled ===
  false`, slices with `args.fallback`, and (W4) transport outliers count.
- UI: a bar under the toolbar (hidden when empty) listing each warning with
  a count and a `show` link that sets the search filter or selects the first
  offending slice. Fallback slices are drawn with a red diagonal hatch
  (`ctx.createPattern` on an offscreen 6x6 canvas, built once per theme) on
  top of their color. Wall time tab gets a `Fallbacks` row: count and time.
- Sample: `make_sample_trace.py --with-warnings` makes `layer1`'s
  `gate_up_proj` run on CPU (`host.cpu`, `args.fallback = "htp_disabled"`,
  no DSP events) and sets `dropped = {host: 0, dsp: 137}`,
  `clock_sync.violations = 2`.
- Acceptance: the banner lists 3 items on that sample; the hatch is visible
  at the fitted zoom; `summarize.py` reports the same counts.

### W8. Tokens strip (M)

- Goal: per-token cost as kv grows.
- Data: from `host.phase` slices: per phase `wall`, `calls` (count of
  `host.wait` inside), `bytes` (sum of seam bytes inside), `transport` (sum
  of seam dur), `dsp_busy` (`metrics(range).any`), `cpu`. TTFT = the
  `prefill` phase wall (document that it excludes tokenization until the
  app emits a `tokenize` span).
- UI: new `Tokens` tab: inline SVG bars in token order, stacked
  CPU / transport / DSP busy / other, with a line for calls per token on a
  second small chart (no dual axis: two charts). Hover tooltip, click ->
  zoom to that token. Header numbers: TTFT, median ms/token, calls/token.
- Sample: `--tokens N` (default 8); attention time grows linearly with kv
  (+2 % per token) so the trend is visible.
- Acceptance: bars strictly increase in the sample; clicking token 3 sets
  the view to its span.

### W9. QNN optrace import (L; data-dependent)

- Goal: a vendor optrace on the same screen as ours.
- Converter `qnn_optrace_to_nntr.py in.json -o out.json [--clock-mhz 1200]
  [--pid 3]`. **Check the real file first** and record findings in the
  script's docstring: the `ts` unit (ref_16 quotes cycles at 1.2 GHz; if
  `ts` is cycles convert to µs), the event `args` keys that carry tile shape
  and cycle counts (`Cycles per Packet` is named in ref_16 §3.3), the tid
  layout (256 DMA, 512-519 HVX, 768 HMX, tid 1 non-executed views). Map
  cats by tid (`dsp.hmx | dsp.hvx | dsp.dma`), drop tid 1 by default
  (`--keep-views` to keep them as `dsp.sync`), set `args.engine`, derive
  `args.elems` as the product of the tile shape when present, keep the
  original name, add `args.op` from the node name prefix (`node_linear_2`).
- Viewer: `Add trace...` reuses the W3 multi-model loader (pid offset so
  the QNN process lands under ours). Kernels tab: a `QNN ref cy/elem`
  column from a built-in table (ref_16 §3.3: HMX conv 0.01-0.06, elementwise
  0.07-0.10, softmax 0.24, transpose 0.71, pathological 1.52) keyed by
  `args.class` which the generator and the converter both set
  (`matmul | elementwise | softmax | transpose | quant | dequant | dma`).
- Acceptance: the converter round-trips a hand-written 10-event fixture in
  `test/fixtures/qnn_mini.json` (write it from the field names found on the
  real file); the viewer shows it as `pid 3` with HMX at tid 768.

### W10. Minimap, collapsing, search navigation (M)

- Minimap: a 36 px canvas strip between the toolbar and the ruler. Per
  pixel column: busy fraction of DSP lanes (top half) and host threads
  (bottom half), gray; the viewport as a rectangle; drag moves it, click
  centers, wheel zooms as on the main canvas. Computed once per load from
  the unit-busy series (W1), not per frame.
- Collapsing: click a process header toggles it (rows rebuilt via
  `layoutRows`); double-click a thread label collapses that thread to one
  row (depth 1, deeper slices hidden). State kept in `opts.collapsed`.
- Search: `Enter` jumps to the next match in ts order (wraps),
  `Shift+Enter` previous, `f` fits the view to the current match, the input
  shows `k / n`. Matches are the events passing the existing `matches()`.
- Acceptance: with filter `softmax`, `Enter` x3 selects three different
  slices in increasing ts; collapsing `HTP` removes its rows and restores
  them on the second click.

### W11. URL hash state (S)

- State: `{t: trace key, v: [t0, t1] relative to tmin (µs, rounded), s:
  selected slice as "pid:tid:index", c: colorBy, tab, r: range as
  "phase:i" | "sel:t0:t1"}` encoded as `key=value&...` in `location.hash`
  via `history.replaceState`, debounced 200 ms. Applied on load after the
  trace is parsed; unknown keys ignored; a hash for a trace not embedded is
  ignored with a console warning.
- Acceptance: reloading the page with a hash restores zoom, selection and
  tab; `check.js` covers it by navigating to `#...` directly.

### W12. Export + `summarize.py` (M)

- Viewer: `Copy CSV` on Kernels, Layers, Tokens, Transport tables (clipboard;
  the artifact sandbox blocks downloads, so also offer `Download` which
  works from `file://`) and `Copy metrics JSON` in the Wall time tab.
- JSON schema `nntr_trace.metrics.v1` (document in README):
  ```
  { "schema": "nntr_trace.metrics.v1", "range": {"label", "t0_us", "t1_us"},
    "wall_us", "buckets": {"cpu","rpc","hmx_only","overlap","hvx_only",
    "dma_only","dsp_idle","load","other"}, "compression", "idle_ratio",
    "engines": [{"track","busy"}], "kernels": [{"engine","name","calls",
    "total_us","mean_us","elems","cy_per_elem","tops"}], "layers": [...],
    "tokens": [...], "transport": {"n","fixed_us","us_per_mb","outliers"},
    "pool_tail": [...], "warnings": {...} }
  ```
- `summarize.py trace.json [--range phase:N|all|t0:t1] [-o metrics.json]`:
  a port of `metrics / kernels / tokens / transport / pool_tail / warnings`
  with the same interval primitives, stdlib only, exit code 1 on
  `--fail-if compression<1.5,idle_ratio>0.05,...` (the P5 gate).
- Acceptance: `test_summarize.py` compares `summarize.py` output with
  `__nntr.metricsJSON()` dumped by `check.js` on the same sample: every
  number within 1e-6 relative (both implement the same definitions).

### W13. Counter budgets and hi-water (S)

- `metadata.budgets = {"VTCM (KB)": 8192}` (generator writes it). Counter
  rows draw the budget as a hairline at the budget value and scale to
  `max(budget, max value)`; the hi-water point is marked and its value
  printed at the right edge of the row. Hover on a counter row shows the
  value at the cursor.
- Acceptance: the sample's VTCM row shows the 8192 line above all bars.

## 2. Order and batches

| batch | items | why this order | PR title |
|---|---|---|---|
| A | W0, W1, W2, W7, W13 | foundation + the three highest-value reads; nothing depends on device data | `[tools] nntr_trace viewer: unit-busy strip, range selection, warnings` |
| B | W5, W6, W4, W8, W12 | analysis tabs that share the metrics code and produce `summarize.py` | `[tools] nntr_trace: pool tail, transport fit, tokens, metrics export` |
| C | W10, W11 | navigation and sharing; touches `layoutRows` from W0 only | `[tools] nntr_trace viewer: minimap, collapsing, search navigation, hash state` |
| D | W3, W9 | multi-model; W9 needs a real QNN optrace to finish (leave the converter behind a fixture until one is available) | `[tools] nntr_trace: A/B compare and QNN optrace import` |

Dependencies: W1 -> W10 (minimap uses the busy series); W3 -> W9 (loader);
W12 depends on W4, W5, W8 for its sections but can land with those sections
empty; W2 is independent but W4/W8 should use `range` so they benefit.

Within a batch, one commit per item (W0's refactor is its own commit before
W1). Update `README.md` in the last commit of each batch.

## 3. Definition of done (per batch)

1. `test/run.sh` passes: samples generated, page bundled, `check.js` green
   (no page errors, assertions for every item in the batch), `python3 -m
   unittest discover tools/nntr_trace/test`.
2. Both themes checked once by screenshot at 1400 px and 400 px widths
   (attach to the PR).
3. `README.md` lists the new controls and the metrics schema version.
4. Commit messages carry the test command and result.

## 4. Non-goals

- No framework, no bundler, no external CDN dependency, no WebGL. The
  canvas renderer is fast enough for ~100k slices; if a real trace exceeds
  that, cull by pixel bucket in `draw()` (coalesce slices narrower than
  0.3 px per row) before considering anything else.
- No runtime instrumentation here: P1+ of the plan document lives in
  `nntrainer/` and its own PRs. This directory only consumes `trace.json`.
