#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render the HTP attention device run as tables, and optionally as HTML.

Reads the ``*_FIELD`` / ``ATTN_STAGE`` marker lines the device tests emit and
turns them into something readable. The markers are keyed by name rather than
by column position on purpose: a report script keyed by position silently
printed a stale number for every shape past the first, twice, in this project
(MHA_HTP_PLAN.md 9.7).

Usage
    tools/htp_attn_report.py                       # reads the default /tmp logs
    tools/htp_attn_report.py run.log               # or any files you name
    tools/htp_attn_report.py --html report.html    # also write a HTML report

Stdlib only: this has to run on whatever machine the device is plugged into.
"""

import argparse
import html
import os
import re
import sys

DEFAULT_LOGS = [
    "/tmp/hvx_mm_u8i4_device_run.log",
    "/tmp/hvx_softmax_device_run.log",
    "/tmp/hvx_attn_device_run.log",
]

MARKER = re.compile(
    r"^(?P<kind>[A-Z0-9_]+_FIELD|ATTN_STAGE)\s+path=(?P<path>\S+)\s+"
    r"field=(?P<field>\S+)\s+value=(?P<value>\S+)\s*$"
)

WIDTHS = ["I8I8", "I4I4", "I8I4", "I4I8"]
WIDTH_LABEL = {"I8I8": "I8,I8", "I4I4": "I4,I4", "I8I4": "I8,I4", "I4I8": "I4,I8"}
# qk and pv are the two layer_run consumers; the rest are ours.
STAGES = ["qk_us", "softmax_us", "pv_us", "accum_us", "gather_us"]
STAGE_LABEL = {
    "qk_us": "Q.Kt",
    "softmax_us": "softmax",
    "pv_us": "P.V",
    "accum_us": "f32 accum",
    "gather_us": "gather+1/l",
}


def parse(paths):
    """-> {(kind, path, field): value}, values kept as float where possible."""
    out = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"  (skipping missing {p})", file=sys.stderr)
            continue
        with open(p, errors="replace") as fh:
            for line in fh:
                m = MARKER.match(line.strip())
                if not m:
                    continue
                try:
                    v = float(m["value"])
                except ValueError:
                    v = m["value"]
                out[(m["kind"], m["path"], m["field"])] = v
    return out


def get(rec, kind, path, field, default=None):
    return rec.get((kind, path, field), default)


def shapes(rec):
    """Which (regime, kv_len) the forward timing covered, in run order."""
    seen = []
    for (kind, path, _f) in rec:
        if kind != "ATTN_FIELD" or not path.startswith("forward_"):
            continue
        m = re.match(r"forward_(dec|pre)_kv(\d+)_", path)
        if m and (m[1], int(m[2])) not in seen:
            seen.append((m[1], int(m[2])))
    return sorted(seen, key=lambda t: (t[1], t[0]))


def fmt(v, nd=1):
    if v is None:
        return "-"
    if isinstance(v, str):
        return v
    return f"{v:,.{nd}f}"


def table(rows, headers, align=None):
    """Fixed-width text table. Returns a list of lines."""
    cols = len(headers)
    align = align or ["l"] + ["r"] * (cols - 1)
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(str(c)))
    def line(cells):
        parts = []
        for i, c in enumerate(cells):
            parts.append(str(c).ljust(w[i]) if align[i] == "l"
                         else str(c).rjust(w[i]))
        return "  " + "  ".join(parts)
    out = [line(headers), "  " + "  ".join("-" * x for x in w)]
    out += [line(r) for r in rows]
    return out


def section(title):
    return ["", f"== {title} " + "=" * max(0, 68 - len(title)), ""]


def correctness(rec):
    """The gates, so a green run is visible at a glance rather than scrolled."""
    checks = [
        ("S band vs host model, 384 cases", "ATTN_FIELD", "scores",
         "max_rel_err", "0 means bit-identical"),
        ("  reference dynamic range", "ATTN_FIELD", "scores",
         "min_ref_dynamic_range", "> 0, else the 0 above is vacuous"),
        ("  V width never reaches S", "ATTN_FIELD", "scores",
         "v_width_invariant", "1 = bitwise invariant"),
        ("  lifecycle repeatable", "ATTN_FIELD", "scores",
         "lifecycle_repeatable", "1 = re-register reproduces"),
        ("fused forward vs host model", "ATTN_FIELD", "forward",
         "max_rel_err", "tolerance 5e-3 at (I8,I8)"),
        ("blocked softmax, 756 cases: p", "BLOCKED_FIELD", "softmax_blocked",
         "max_abs_err_p", "<= 1e-6"),
        ("  l summation-order margin", "BLOCKED_FIELD", "softmax_blocked",
         "l_sum_order_margin", "<= 1.0"),
    ]
    rows = []
    for label, kind, path, field, note in checks:
        v = get(rec, kind, path, field)
        rows.append([label, "-" if v is None else f"{v:g}", note])
    return table(rows, ["check", "value", "expected"], ["l", "r", "l"])


def per_layer(rec):
    rows = []
    for regime, kv in shapes(rec):
        row = [f"{kv}", "decode" if regime == "dec" else "prefill"]
        for wd in WIDTHS:
            row.append(fmt(get(rec, "ATTN_FIELD", f"forward_{regime}_kv{kv}_{wd}",
                               "us_per_layer")))
        tok = get(rec, "ATTN_FIELD", f"forward_{regime}_kv{kv}_I8I8",
                  "us_per_token_all_layers")
        row.append("-" if tok is None else f"{tok / 1000.0:,.1f}")
        rows.append(row)
    return table(rows,
                 ["kv_len", "regime"] + [WIDTH_LABEL[w] for w in WIDTHS]
                 + ["28 layers (ms)"])


def breakdown(rec, width="I8I8"):
    """Where the DSP-internal time actually goes, per stage, with shares."""
    rows = []
    for regime, kv in shapes(rec):
        path = f"forward_{regime}_kv{kv}_{width}"
        vals = {s: get(rec, "ATTN_STAGE", path, s) for s in STAGES}
        total = get(rec, "ATTN_STAGE", path, "dsp_total_us")
        wall = get(rec, "ATTN_FIELD", path, "us_per_layer")
        transport = get(rec, "ATTN_STAGE", path, "transport_us")
        calls = get(rec, "ATTN_STAGE", path, "layer_run_calls")
        if total is None:
            continue
        row = [f"{kv}", "decode" if regime == "dec" else "prefill"]
        for s in STAGES:
            v = vals[s]
            pct = (100.0 * v / total) if (v is not None and total) else None
            row.append("-" if v is None else f"{v:,.0f} ({pct:.0f}%)")
        row += [fmt(total, 0), fmt(transport, 0), fmt(wall, 0), fmt(calls, 0)]
        rows.append(row)
    if not rows:
        return ["  (no ATTN_STAGE lines -- run a build that has "
                "attn_forward_timed)"]
    return table(rows,
                 ["kv_len", "regime"] + [STAGE_LABEL[s] for s in STAGES]
                 + ["dsp total", "transport", "wall", "layer_run"])


def cost_model(rec, width="I8I8"):
    """Total time against layer_run call count.

    If the per-call figure is flat across shapes, the cost is dominated by how
    many times layer_run is entered rather than by the arithmetic inside it --
    which is the malloc storm MHA_HTP_U8_TASKS.md 1.8 predicted.
    """
    rows = []
    for regime, kv in shapes(rec):
        path = f"forward_{regime}_kv{kv}_{width}"
        wall = get(rec, "ATTN_FIELD", path, "us_per_layer")
        calls = get(rec, "ATTN_STAGE", path, "layer_run_calls")
        if not wall or not calls:
            continue
        rows.append([f"{kv}", "decode" if regime == "dec" else "prefill",
                     fmt(calls, 0), fmt(wall, 0), fmt(wall / calls, 0)])
    if not rows:
        return ["  (needs ATTN_STAGE layer_run_calls)"]
    return table(rows, ["kv_len", "regime", "layer_run calls", "wall us",
                        "us per call"])


def bar_svg(labels, values, colors, width=760, row_h=26):
    """One stacked bar per row, pure SVG so the report needs no JS or CDN."""
    if not values:
        return ""
    total_max = max(sum(v) for v in values) or 1.0
    h = row_h * len(labels) + 30
    parts = [f'<svg viewBox="0 0 {width} {h}" width="100%" '
             f'style="max-width:{width}px" role="img">']
    for i, (lab, vs) in enumerate(zip(labels, values)):
        y = i * row_h + 4
        x = 150.0
        parts.append(f'<text x="0" y="{y + 14}" font-size="12" '
                     f'fill="currentColor">{html.escape(lab)}</text>')
        for v, c in zip(vs, colors):
            w = (width - 170.0) * v / total_max
            if w > 0.3:
                parts.append(f'<rect x="{x:.1f}" y="{y}" width="{w:.1f}" '
                             f'height="16" fill="{c}"><title>{v:,.0f} us'
                             f'</title></rect>')
            x += w
    parts.append("</svg>")
    return "".join(parts)


def write_html(rec, path):
    colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#b279a2"]
    labels, values = [], []
    for regime, kv in shapes(rec):
        p = f"forward_{regime}_kv{kv}_I8I8"
        vs = [get(rec, "ATTN_STAGE", p, s) or 0.0 for s in STAGES]
        if sum(vs) == 0:
            continue
        labels.append(f"kv={kv} {'decode' if regime == 'dec' else 'prefill'}")
        values.append(vs)

    legend = " ".join(
        f'<span style="color:{c}">&#9632;</span> {html.escape(STAGE_LABEL[s])}'
        for s, c in zip(STAGES, colors))

    def pre(lines):
        return "<pre>" + html.escape("\n".join(lines)) + "</pre>"

    body = f"""<title>HTP attention device report</title>
<style>
:root {{ color-scheme: light dark; }}
body {{ font: 14px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
        margin: 2rem auto; max-width: 900px; padding: 0 1rem; }}
h1 {{ font-size: 1.3rem; }} h2 {{ font-size: 1rem; margin-top: 2rem; }}
pre {{ overflow-x: auto; padding: .75rem; border-radius: 6px;
       background: rgba(127,127,127,.12); }}
</style>
<h1>HTP attention &mdash; device run</h1>
<h2>Correctness gates</h2>{pre(correctness(rec))}
<h2>Cost per fused layer (us, includes FastRPC transport)</h2>
{pre(per_layer(rec))}
<h2>Where the DSP time goes (I8,I8)</h2>
<p>{legend}</p>{bar_svg(labels, values, colors)}
{pre(breakdown(rec))}
<h2>Cost against layer_run call count</h2>{pre(cost_model(rec))}
"""
    with open(path, "w") as fh:
        fh.write(body)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", default=None,
                    help="device run logs (default: the three /tmp ones)")
    ap.add_argument("--html", metavar="FILE", help="also write a HTML report")
    ap.add_argument("--width", default="I8I8", choices=WIDTHS,
                    help="width pair for the breakdown tables (default I8I8)")
    args = ap.parse_args()

    rec = parse(args.logs or DEFAULT_LOGS)
    if not rec:
        print("no marker lines found -- did the device run produce logs?",
              file=sys.stderr)
        return 1

    out = []
    out += section("Correctness gates")
    out += correctness(rec)
    out += section("Cost per fused layer, us (includes FastRPC transport)")
    out += per_layer(rec)
    out += section(f"Where the DSP time goes ({WIDTH_LABEL[args.width]})")
    out += breakdown(rec, args.width)
    out += section("Cost against layer_run call count")
    out += cost_model(rec, args.width)
    out += ["", "  A flat 'us per call' means the cost tracks how many times",
            "  layer_run is entered, not the arithmetic inside it.", ""]
    print("\n".join(out))

    if args.html:
        write_html(rec, args.html)
    return 0


if __name__ == "__main__":
    sys.exit(main())
