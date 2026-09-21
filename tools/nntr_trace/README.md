# nntr_trace

Timeline profiling for nntrainer runs that mix CPU, Hexagon HTP (HMX / HVX /
DMA) and QNN layers. The design and the phased plan are in
[`docs/backend_guide/HTP_TRACE_PROFILER.md`](../../docs/backend_guide/HTP_TRACE_PROFILER.md);
the viewer's own roadmap is [`VIEWER_PLAN.md`](VIEWER_PLAN.md). Nothing in the
runtime emits a trace yet: this directory holds the file format by example,
the viewer, and converters for the logs the HTP device tests already print.

| file | what it does |
|---|---|
| `viewer.html` | Standalone timeline viewer in the chrome://tracing idiom: no build, no server, no external scripts. Open it in a browser and press `Load` to pick a `trace.json`. |
| `bundle.py` | Bakes one or more traces into the viewer so a single HTML file carries the data (mail, chat, an artifact link). |
| `make_sample_trace.py` | Emits a synthetic `trace.json` in the target schema, durations scaled from the numbers measured in the HTP branch docs. `--pipelined` projects the doc-35 T2 overlap; `--with-warnings` adds a CPU fallback, dropped records and clock violations; `--tokens N` sets the decode length. |
| `stage_us_to_trace.py` | Converts `FC_STAGE` / `ATTN_STAGE` marker logs from `unittest_hvx_fc` / `unittest_hvx_attn` into the schema, stage totals laid out back to back. |
| `summarize.py` | The viewer's metrics from the command line (`--range`, `--fail-if` gates), held to the viewer's numbers by a test. |
| `qnn_optrace_to_nntr.py` | Rebases a QAIRT HTP optrace (`*_chromeTrace_opTrace.json`) onto the schema as pid 3 so it opens beside ours. Written against a fixture: its docstring lists the field assumptions to verify on a real file. |
| `test/run.sh` | Generates the samples, bundles them, runs `test/check.js` (headless Chromium) and the python unit tests. |

```bash
python3 tools/nntr_trace/make_sample_trace.py            # trace_asbuilt.json
python3 tools/nntr_trace/make_sample_trace.py --pipelined
python3 tools/nntr_trace/stage_us_to_trace.py /tmp/hvx_fc_device_run.log -o fc.json
python3 tools/nntr_trace/bundle.py -o report.html --trace "run 1=trace_asbuilt.json"
# then open tools/nntr_trace/viewer.html (Load) or report.html,
# or drop the json on https://ui.perfetto.dev
bash tools/nntr_trace/test/run.sh                         # before committing
```

## Capturing from a device run

Two ways, both ending in the same viewer.

**A. No rebuild.** Any HTP build already prints a per-shape summary at exit;
that becomes one representative call per shape (averages, no timeline):

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 ./nntrainer_causallm ./models/<model>" 2>&1 | tee run.log
python3 tools/nntr_trace/htp_profile_to_trace.py run.log -o profile.json
python3 tools/nntr_trace/bundle.py -o report.html --trace "run=profile.json"
```

**B. Per-call timeline.** Needs a build with the `HtpTrace` recorder
(`nntrainer/tensor/htp_backend/htp_trace.{h,cpp}`, on the HTP branch).
`NNTR_TRACE` names the output and implies profile level 2 so the timed
FastRPC entries run:

```bash
adb shell "cd /data/local/tmp/nntrainer/causallm && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  NNTR_NUM_THREADS=8 NNTR_TRACE=/data/local/tmp/trace.json ./nntrainer_causallm ./models/<model>"
adb pull /data/local/tmp/trace.json
python3 tools/nntr_trace/bundle.py -o run.html --trace "run=trace.json"
```

Every FastRPC call gets a host wait span, its seam halves and a DSP entry
holding that call's stage totals on their lanes, with prefill and each
decode token marked as phases. Lane overlap (HMX ∥ HVX) reads 0 at this
level: the stage totals are laid out sequentially inside a call, which the
metadata states. Measuring real overlap needs the DSP-side ring of P3 in
the plan document.

## Viewer

- **Timeline.** One process per pid (host, HTP, later QNN), one row per
  thread, nested slices, flow arrows from the FastRPC seam to the DSP entry.
  Every slice label is `op: name`; `Color:` switches between op, name and
  engine. Under each process header a **units busy** strip shows how many
  lanes (HMX, HVX-i, DMA; or CPU threads) are busy at each instant, which is
  parallel compression unrolled over time. Counters (VTCM) draw their
  `metadata.budgets` line and hi-water mark.
- **Navigation.** W/S zoom, A/D pan, `0` or double-click fit, drag to pan,
  wheel to zoom at the cursor. The overview strip under the toolbar shows
  DSP and host busy density with the viewport; drag it, click to center,
  wheel to zoom. Click a process header to collapse it, double-click a
  thread label to fold that thread to one row. Find: Enter / Shift+Enter
  walk the matches in time order (`k / n` beside the box), `f` fits the
  view to the current match, Esc clears. The view, selection, tab, range,
  color mode and filter live in the URL hash, so a zoomed-in finding can be
  shared as a link to the same bundled page.
- **Compare.** `Compare:` loads a second trace (an embedded one or a file)
  as B under A; `View:` shows A, B or both. The Compare tab puts the
  wall-time buckets, parallel compression, idle and transport share side
  by side with deltas, joins the kernels on `engine + op: name` (calls,
  total, mean, cy/elem, Δ), and the layers on name + phase. A converted QNN
  optrace loads the same way (pid 3); the Kernels tab's `QNN ref` column
  shows ref_16's cycles/element for each kernel class.
- **Range.** Drag in the ruler (or shift+drag) to select a time range, `m`
  to make the selected slice's span the range, `Esc` to clear; `Range:` also
  offers each prefill / decode phase. Every analysis tab is computed over
  the range.
- **Warnings banner.** CPU fallbacks (`args.fallback`, drawn hatched),
  dropped records (`metadata.dropped`), clock-sync violations and a disabled
  HTP backend are listed under the toolbar.
- **Analysis tabs.** Selection (title, track, duration, cy/elem, MACs, TOPS
  and share of `metadata.hmx_peak_tops` when known, args), Wall time
  (buckets: CPU, FastRPC seam, HMX only, HMX ∥ HVX, HVX only, DMA only, DSP
  idle, load, other; parallel compression; idle inside DSP calls; CPU
  fallbacks; copy the metrics JSON), Engines (busy per track, and the HVX
  pool tail per fork-join kind against the `units ≥ threads × 4` rule),
  Kernels (`op: name` rows with cycles/element, TOPS and outlier flags;
  click to filter), Layers (per-layer stack and numbers; click to zoom),
  Transport (FastRPC seam time against payload bytes, least-squares
  fixed + per-MB fit, outliers beyond 2σ; click to select the call), Tokens
  (per-token bars stacked by CPU / FastRPC / DSP busy / other, calls per
  token, TTFT and median token time; click to zoom). Tables have Copy and
  Download CSV buttons.

## summarize.py

The same metrics from the command line, for the device gate:

```bash
python3 tools/nntr_trace/summarize.py trace.json -o metrics.json
python3 tools/nntr_trace/summarize.py trace.json --range phase:1
python3 tools/nntr_trace/summarize.py trace.json --fail-if 'compression<1.5' --fail-if 'idle_ratio>0.05'
```

`test/test_summarize.py` holds it to the viewer's numbers (1e-6 relative)
on the sample, so the two cannot drift. Schema `nntr_trace.metrics.v1`:
`range`, `wall_us`, `buckets`, `compression`, `idle_ratio`, `engines`,
`kernels`, `layers`, `tokens`, `token_summary`, `transport`, `pool_tail`,
`warnings`, `hmx_peak_tops`.

## Format

Chrome Trace Event Format (`X` spans, `M` names, `s`/`f` flows, `C`
counters). DSP thread ids follow the QNN optrace convention (DMA 256, HVX
512.., HMX 768) so a vendor optrace and ours line up side by side. Compute
spans carry `args.engine`, `args.elems` and (when the tracer has them)
`args.cycles`; `metadata.dsp_clock_mhz` is the fallback for cycles/element.
The metrics the viewer derives are exposed as `window.__nntr.metricsJSON()`
(schema `nntr_trace.metrics.v1`).

Stdlib only, on purpose: these run on whatever machine the device is
plugged into.
