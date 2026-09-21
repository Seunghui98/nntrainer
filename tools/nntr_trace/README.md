# nntr_trace

Timeline profiling for nntrainer runs that mix CPU, Hexagon HTP (HMX / HVX /
DMA) and QNN layers. The design and the phased plan are in
[`docs/backend_guide/HTP_TRACE_PROFILER.md`](../../docs/backend_guide/HTP_TRACE_PROFILER.md).
This directory holds the P0 pieces: the file format by example, a viewer, and
a converter for the logs the HTP device tests already print.

| file | what it does |
|---|---|
| `viewer.html` | Standalone timeline viewer (no build, no server). Open it in a browser and load a `trace.json`; wheel zooms, drag pans, click selects. Below the timeline it derives engine busy %, parallel compression, DSP idle, FastRPC transport share, cycles/element per kernel, and a per-layer breakdown. |
| `make_sample_trace.py` | Emits a synthetic `trace.json` in the schema the real tracer will write, with durations scaled from the numbers measured in the HTP branch docs. `--pipelined` emits the doc-35 T2 projection (HMX and HVX overlapping) for comparison. |
| `stage_us_to_trace.py` | Converts `FC_STAGE` / `ATTN_STAGE` marker logs from `unittest_hvx_fc` / `unittest_hvx_attn` into the schema, stage totals laid out back to back. |

```bash
python3 tools/nntr_trace/make_sample_trace.py            # trace_asbuilt.json
python3 tools/nntr_trace/make_sample_trace.py --pipelined
python3 tools/nntr_trace/stage_us_to_trace.py /tmp/hvx_fc_device_run.log -o fc.json
# then open tools/nntr_trace/viewer.html and press "Open trace.json",
# or drop the file on https://ui.perfetto.dev
```

The format is the Chrome Trace Event Format (`X` spans, `M` names, `s`/`f`
flows, `C` counters). DSP thread ids follow the QNN optrace convention
(DMA 256, HVX 512.., HMX 768) so a vendor optrace and ours line up side by
side. Compute spans carry `args.engine` and `args.elems`; the viewer needs
those two and `metadata.dsp_clock_mhz` (or `args.cycles`) for cycles/element.

Stdlib only, on purpose: these run on whatever machine the device is
plugged into.
