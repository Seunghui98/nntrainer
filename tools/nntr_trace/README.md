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

## Viewer

- **Timeline.** One process per pid (host, HTP, later QNN), one row per
  thread, nested slices, flow arrows from the FastRPC seam to the DSP entry.
  Every slice label is `op: name`; `Color:` switches between op, name and
  engine. Under each process header a **units busy** strip shows how many
  lanes (HMX, HVX-i, DMA; or CPU threads) are busy at each instant, which is
  parallel compression unrolled over time. Counters (VTCM) draw their
  `metadata.budgets` line and hi-water mark.
- **Navigation.** W/S zoom, A/D pan, `0` or double-click fit, drag to pan,
  wheel to zoom at the cursor, click a process header to collapse it.
- **Range.** Drag in the ruler (or shift+drag) to select a time range, `m`
  to make the selected slice's span the range, `Esc` to clear; `Range:` also
  offers each prefill / decode phase. Every analysis tab is computed over
  the range.
- **Warnings banner.** CPU fallbacks (`args.fallback`, drawn hatched),
  dropped records (`metadata.dropped`), clock-sync violations and a disabled
  HTP backend are listed under the toolbar.
- **Analysis tabs.** Selection (title, track, duration, cy/elem, args),
  Wall time (buckets: CPU, FastRPC seam, HMX only, HMX ∥ HVX, HVX only, DMA
  only, DSP idle, load, other; parallel compression; idle inside DSP calls;
  CPU fallbacks), Engines (busy per track), Kernels (`op: name` rows with
  cycles/element and outlier flags; click to filter), Layers (per-layer
  stack; click to zoom).

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
