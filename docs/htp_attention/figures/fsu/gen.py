#!/usr/bin/env python3
"""Generates the three expert-streaming figures (doc 57) as 1920x1080 HTML
pages; render.js screenshots them to PNG. Numbers: docs 52, 53 and 55."""

HEAD = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=1920">
<title>{title}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=block" rel="stylesheet">
<style>
:root {{
  --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --base:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --npu:#2a78d6; --npu-t:#cde2fb; --read:#eb6834; --read-t:#fbe0d4;
  --host:#1baf7a; --host-t:#d3f1e6; --crit:#d03b3b; --crit-t:#f7dcdc;
  --idle:#e9e8e3;
}}
html,body {{ margin:0; background:var(--surface); }}
svg {{ display:block; font-family:"Noto Sans KR",system-ui,sans-serif; }}
.t1 {{ font-size:44px; font-weight:700; fill:var(--ink); }}
.t2 {{ font-size:24px; fill:var(--ink2); }}
.lane {{ font-size:22px; font-weight:700; fill:var(--ink); }}
.lane2 {{ font-size:17px; fill:var(--ink2); }}
.lbl {{ font-size:19px; fill:var(--ink); }}
.lblw {{ font-size:19px; font-weight:700; fill:#fff; }}
.sm {{ font-size:16px; fill:var(--ink2); }}
.xs {{ font-size:15px; fill:var(--muted); }}
.big {{ font-size:46px; font-weight:700; fill:var(--ink); }}
.h3 {{ font-size:22px; font-weight:700; fill:var(--ink); }}
</style></head><body>
<svg width="1920" height="1080" viewBox="0 0 1920 1080" role="img" aria-label="{title}">
<defs>
 <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>
 <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#eb6834"/></marker>
</defs>
<rect width="1920" height="1080" fill="var(--surface)"/>
"""
TAIL = "</svg></body></html>\n"


def rect(x, y, w, h, fill, rx=4, stroke=None, extra=""):
    s = f' stroke="{stroke}" stroke-width="2"' if stroke else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}"{s} {extra}/>')


def text(x, y, s, cls="lbl", anchor="start", extra=""):
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}" {extra}>{s}</text>'


def card(x, y, w, h, accent, title, big, lines, icon=""):
    out = [rect(x, y, w, h, "#fff", 10, None,
                'style="stroke:var(--ring);stroke-width:2"'),
           rect(x, y, 8, h, accent, 4),
           text(x + 30, y + 44, icon + title, "h3"),
           text(x + 30, y + 108, big, "big")]
    for i, ln in enumerate(lines):
        out.append(text(x + 30, y + 150 + 30 * i, ln, "sm"))
    return "\n".join(out)


# ---------------------------------------------------------------- figure 1
def fig1():
    o = [HEAD.format(title="expert 가중치 위치")]
    o.append(text(80, 92, "Gemma-4 26B-A4B: expert는 flash에 두고, 쓸 만큼만 NPU 메모리로", "t1"))
    o.append(text(80, 140, "expert 3,840개(30층 × 128) = 11.6 GB 중 DRAM에는 480칸(층당 C=16) ≈ 1.4 GB만 둔다. "
                          "나머지는 필요할 때 flash에서 읽는다", "t2"))
    # column frames
    cols = [(80, "Flash (UFS)", "모델 파일 약 12–13 GB · 실측 3.0 GB/s"),
            (700, "DRAM: DSP가 매핑한 ION 아레나", "CPU가 쓰고, NPU가 offset으로 읽는다"),
            (1400, "NPU (Hexagon V79)", "HMX 행렬 · HVX 벡터")]
    widths = [520, 600, 440]
    for (x, h, s), w in zip(cols, widths):
        o.append(rect(x, 190, w, 640, "#fff", 12, None, 'style="stroke:var(--ring);stroke-width:2"'))
        o.append(text(x + 28, 238, h, "lane"))
        o.append(text(x + 28, 268, s, "lane2"))
    # flash: experts block + non-expert block
    o.append(rect(108, 300, 464, 400, "var(--read-t)", 8))
    o.append(text(130, 340, "expert 3,840개 · 11.6 GB", "h3"))
    o.append(text(130, 372, "1개 2.87 MiB, QS4CX_WH", "sm"))
    o.append(text(130, 398, "(HMX 타일 순서로 미리 변환)", "sm"))
    # small grid suggesting many experts
    for r in range(10):
        for c in range(26):
            o.append(rect(130 + c * 16.6, 420 + r * 26, 13, 20, "var(--read)", 2,
                          None, 'opacity="0.55"'))
    o.append(rect(108, 716, 464, 90, "var(--idle)", 8))
    o.append(text(130, 752, "attention · dense FFN (QS4CX)", "lbl"))
    o.append(text(130, 782, "embedding = lm_head (Q4_0, tie)", "lbl"))
    # DRAM: slot pool 30 x 16
    o.append(text(728, 318, "expert 칸 풀 480칸 (30층이 함께 쓰는 LRU)", "h3"))
    gx, gy, cw, ch = 728, 336, 33.5, 13
    states = {}
    for r in range(30):
        for c in range(16):
            # illustrative snapshot: in use / ready / being read / free
            k = r * 16 + c
            st = ("npu" if k < 128 else "ready" if k < 256 else
                  "read" if k < 384 else "idle")
            fill = {"npu": "var(--npu)", "ready": "var(--npu-t)",
                    "read": "var(--read)", "idle": "var(--idle)"}[st]
            o.append(rect(gx + c * cw, gy + r * ch, cw - 2, ch - 2, fill, 2))
    ly = 742
    for i, (f, s) in enumerate([("var(--npu)", "지금 층이 쓰는 중"),
                                ("var(--npu-t)", "다음 층 것, 읽기 끝남"),
                                ("var(--read)", "그 다음 층 것, 읽는 중"),
                                ("var(--idle)", "비어 있음 / 지난 층")]):
        x = 728 + (i % 2) * 270
        y = ly + (i // 2) * 34
        o.append(rect(x, y - 16, 20, 20, f, 3))
        o.append(text(x + 30, y, s, "sm"))
    o.append(text(728, 822, "그 밖에 상주: attention·FFN 가중치, lm_head, KV cache", "xs"))
    # NPU
    o.append(rect(1428, 300, 384, 150, "var(--npu-t)", 8))
    o.append(text(1452, 342, "층 L의 MoE 호출", "h3"))
    o.append(text(1452, 376, "칸 offset으로 가중치를", "sm"))
    o.append(text(1452, 402, "DMA로 VTCM에 올려 계산", "sm"))
    o.append(rect(1428, 480, 384, 150, "var(--host-t)", 8))
    o.append(text(1452, 522, "가중치 표 (DSP 쪽)", "h3"))
    o.append(text(1452, 556, "읽어 둔 칸을 offset만으로", "sm"))
    o.append(text(1452, 582, "등록: RPC 한 번에 배치째", "sm"))
    # arrows
    o.append('<path d="M572,520 C640,520 650,560 718,560" stroke="#eb6834" stroke-width="4" fill="none" marker-end="url(#ahr)"/>')
    o.append(text(576, 488, "pread", "lbl", extra='font-weight="700" fill="#eb6834"'))
    o.append(text(576, 600, "리더 스레드 4개", "sm"))
    o.append('<path d="M1270,420 C1330,420 1350,375 1418,375" stroke="#52514e" stroke-width="3" fill="none" marker-end="url(#ah)"/>')
    o.append(text(1290, 470, "DMA", "lbl", extra='font-weight="700"'))
    o.append('<path d="M1270,620 C1330,620 1350,560 1418,560" stroke="#52514e" stroke-width="3" fill="none" marker-end="url(#ah)"/>')
    o.append(text(1290, 655, "offset 등록", "lbl"))
    # footer
    o.append(text(80, 900, "nntrainer의 일반 FSU(config의 fsu: true, 층 단위로 가중치를 내렸다 올림)와는 다른 경로다. "
                          "MoE 층이 expert 단위로 캐시와 선읽기를 직접 관리한다.", "sm"))
    o.append(text(80, 932, "C(층당 칸 수)는 메모리 예산이 정한다: C=16이면 아레나 1.4 GiB, 상주(C=128)면 10.8 GiB라 모바일에서 불가능.", "sm"))
    o.append(text(80, 1030, "출처: 문서 55 §3.2(크기 산술), 53 §3(LFM2 실측), 52 §10.20–10.33(선읽기 설계). "
                           "칸 그림은 한 순간의 예시 상태.", "xs"))
    o.append(TAIL)
    return "\n".join(o)


# ---------------------------------------------------------------- figure 2
def fig2():
    o = [HEAD.format(title="prefill expert 선읽기")]
    o.append(text(80, 92, "prefill: 층 L을 계산하는 동안 뒤 층의 expert를 flash에서 미리 읽는다", "t1"))
    o.append(text(80, 140, "프롬프트 수백 토큰이 top-8로 퍼지면 층마다 128 expert가 거의 다 쓰인다. "
                          "그래서 무엇을 읽을지 그 층의 라우터보다 먼저 안다", "t2"))
    x0, bw, gap = 330, 228, 10
    names = ["층 L", "L+1", "L+2", "L+3", "L+4", "L+5"]
    lanes = [("NPU", "HMX · HVX", 205), ("ARM", "자기 배치 offset 등록", 315),
             ("flash 리더 ×4", "pread → 칸", 385), ("칸 풀 480칸", "그 순간의 구성", 545)]
    for name, sub, y in lanes:
        o.append(text(80, y + 30, name, "lane"))
        o.append(text(80, y + 56, sub, "lane2"))
    for i, nm in enumerate(names):
        x = x0 + i * (bw + gap)
        # NPU block: the layer's calls
        o.append(rect(x + 22, 205, bw - 22, 76, "var(--npu)", 6))
        o.append(text(x + 36, 238, f"{nm} 계산", "lblw"))
        o.append(text(x + 36, 266, "qkv·attn·FFN·MoE", "lblw", extra='style="font-weight:400;font-size:16px"'))
        # ARM register tick before the layer's calls
        o.append(rect(x, 315, 18, 46, "var(--host)", 3))
        # reader bar: the batch two layers ahead, 128 ms of a ~167 ms layer
        rw = round((bw - 22) * 128 / 167)
        nxt = ["L+2", "L+3", "L+4", "L+5", "L+6", "L+7"][i]
        o.append(rect(x + 22, 385, rw, 76, "var(--read)", 6))
        o.append(text(x + 36, 418, f"{nxt} 배치", "lblw"))
        o.append(text(x + 36, 446, "128개 읽기", "lblw", extra='style="font-weight:400;font-size:16px"'))
        # pool composition at this layer: in use / ready / being read / free
        segs = [("var(--npu)", 128), ("var(--npu-t)", 128), ("var(--read)", 128), ("var(--idle)", 96)]
        px, pw = x + 22, bw - 22
        for f, n in segs:
            w = pw * n / 480
            o.append(rect(px, 545, w - 2, 56, f, 3))
            px += w
    # the pool legend, one row under the pool lane
    for i, (f, lb) in enumerate([("var(--npu)", "지금 층이 쓰는 중"),
                                 ("var(--npu-t)", "다음 층 것, 읽기 끝남"),
                                 ("var(--read)", "그 다음 층 것, 읽는 중"),
                                 ("var(--idle)", "빈 칸 (끝난 층에서 비움)")]):
        lx = x0 + 22 + i * 300
        o.append(rect(lx, 618, 18, 18, f, 3))
        o.append(text(lx + 28, 633, lb, "sm"))
    # the example: the L+2 batch read during L reaches L+2's register tick,
    # under the reader lane and up beside the L+2 bar
    xa = x0 + 22 + round((bw - 22) * 128 / 167) / 2
    xt = x0 + 2 * (bw + gap) + 9
    o.append(f'<path d="M{xa},463 L{xa},486 L{xt},486 L{xt},368" stroke="#eb6834" '
             f'stroke-width="3" fill="none" stroke-dasharray="8 6" marker-end="url(#ahr)"/>')
    o.append(text(xa + 14, 516, "두 층 뒤 자기 차례에 받는다: 다 읽혀 있으면 대기 0", "lbl",
                  extra='font-weight="700" fill="#eb6834"'))
    # time axis
    o.append('<path d="M330,668 L1760,668" stroke="#c3c2b7" stroke-width="2" marker-end="url(#ah)"/>')
    o.append(text(1770, 674, "시간", "sm"))
    o.append(text(330, 700, "층 계산 ≈167 ms (prefill 5.01 s ÷ 30층, 실측) · 배치 읽기 ≈128 ms "
                            "(128 × 2.87 MiB ÷ 3.0 GB/s, 산술) → 읽기가 계산 뒤에 숨는다", "sm"))
    o.append(text(330, 726, "층 하나가 끝나면 그 128칸이 비고, 칸이 허용하는 만큼 뒤 층 배치를 줄 세운다(리더는 호출 스레드의 코어를 피해 고정)", "xs"))
    # cards
    o.append(card(80, 762, 560, 230, "var(--npu)", "C=16 (기본): 선읽기 자리 있음",
                  "대기 10 ms", ["선읽기 3,360개 · 동기 읽기 0개", "prefill 5.01 s (89 TPS)"]))
    o.append(card(680, 762, 560, 230, "var(--crit)", "C=5: 칸 150개, 한 층이 거의 다 차지",
                  "대기 7.10 s", ["선읽기 자리 없음 → 동기 읽기 2,893개", "prefill 10.94 s (41 TPS)"], "⚠ "))
    o.append(card(1280, 762, 560, 230, "var(--muted)", "바닥: flash 대역폭",
                  "≈ 3.3 s", ["prefill 1회 ≈10 GB ÷ 3.0 GB/s (산술)", "NPU가 빨라질수록 드러남: 준비율 94→77%"]))
    o.append(text(80, 1040, "실측: Galaxy S25 Ultra(Hexagon V79), 446토큰, MoE만 NPU 구성, 2026-10-02 (문서 55 §10.11–10.12). "
                            "모든 연산 NPU 구성(이번 브랜치)은 기기 미측정. 그림의 막대 길이는 위 두 숫자의 비율.", "xs"))
    o.append(TAIL)
    return "\n".join(o)


# ---------------------------------------------------------------- figure 3
def fig3():
    o = [HEAD.format(title="decode expert 캐시")]
    o.append(text(80, 92, "decode: 라우터가 고른 뒤에야 안다 → LRU 캐시에 남기고, 없으면 그때 읽는다", "t1"))
    o.append(text(80, 140, "토큰 하나는 층마다 128개 중 8개만 쓴다. 어떤 8개인지 라우터 전에는 모르니 "
                          "미리 읽을 수 없고, 칸에 남겨 둔 것이 맞기를 기대한다", "t2"))
    # step boxes
    steps = [(80, "① 라우터", "ARM", ["1행이라 CPU가 더 빠름", "top-8 + 후보 5개"]),
             (500, "② 칸 풀 확인", "ARM", ["8개 중 칸에 있는 것", "= 히트, 없는 것 = 미스"]),
             (920, "③ 미스만 읽기", "flash", ["미스 1개 = 2.87 MiB", "pread → 칸 → 등록"]),
             (1340, "④ MoE 호출", "NPU", ["8 expert로 계산", "칸 offset으로 DMA"])]
    fills = {"NPU": "var(--npu-t)", "ARM": "var(--host-t)", "flash": "var(--read-t)"}
    for x, h, who, lines in steps:
        o.append(rect(x, 200, 380, 200, fills[who], 12))
        o.append(text(x + 26, 248, h, "h3"))
        o.append(text(x + 354, 248, who, "sm", "end"))
        for i, ln in enumerate(lines):
            o.append(text(x + 26, 300 + 34 * i, ln, "lbl"))
        if x < 1340:
            o.append(f'<path d="M{x + 384},300 L{x + 414},300" stroke="#52514e" stroke-width="3" marker-end="url(#ah)"/>')
    # chips: 8 routed experts, 6 hits 2 misses (illustrative)
    o.append(text(80, 470, "예: 한 토큰, 한 층의 top-8", "h3"))
    ids = [17, 3, 88, 41, 120, 9, 64, 102]
    hit = [True, True, False, True, True, True, False, True]
    for i, (e, h) in enumerate(zip(ids, hit)):
        x = 80 + i * 140
        o.append(rect(x, 492, 124, 84, "var(--npu)" if h else "var(--read)", 8))
        o.append(text(x + 62, 530, f"#{e}", "lblw", "middle"))
        o.append(text(x + 62, 560, "히트" if h else "미스", "lblw", "middle",
                      'style="font-weight:400;font-size:16px"'))
    o.append(text(80 + 8 * 140 + 10, 528, "히트 → 바로 계산", "lbl"))
    o.append(text(80 + 8 * 140 + 10, 562, "미스 → flash에서 읽은 뒤 계산 (토큰이 기다린다)", "lbl"))
    # LRU refresh
    o.append(rect(80, 620, 1760, 120, "#fff", 12, None, 'style="stroke:var(--ring);stroke-width:2"'))
    o.append(text(110, 664, "⑤ LRU 갱신", "h3"))
    o.append(text(110, 702, "쓴 8개에 라우터가 다음으로 꼽은 후보 5개까지 \"최근 사용\"으로 올린다. "
                            "가장 오래 안 쓴 칸부터 비워 미스를 채운다", "lbl"))
    # cards
    o.append(card(80, 772, 560, 220, "var(--npu)", "LFM2 실측 (32 expert/층, C=16)",
                  "히트 85%", ["decode 21.7 TPS (상주 24.0)", "C=8이면 히트 57%, 17.9 TPS"]))
    o.append(card(680, 772, 560, 220, "var(--read)", "미스 비용",
                  "2.87 MiB/개", ["÷ 3.0 GB/s ≈ 1 ms + 등록 (산술)", "decode 속도는 미스 수가 정한다"]))
    o.append(card(1280, 772, 560, 220, "var(--muted)", "Gemma (128 expert/층, C=16)",
                  "기기 미측정", ["층당 칸 16 / expert 128", "히트율은 라우팅 분포에 달림"]))
    o.append(text(80, 1040, "출처: LFM2 수치는 문서 53 §3(2026-09-29 실측). 칩의 히트/미스 배치는 예시. "
                            "Gemma decode 히트율과 TPS는 아직 재지 않았다.", "xs"))
    o.append(TAIL)
    return "\n".join(o)


for name, fn in [("fsu_1_placement", fig1), ("fsu_2_prefill_prefetch", fig2),
                 ("fsu_3_decode_cache", fig3)]:
    open(f"{name}.html", "w").write(fn())
print("ok")
