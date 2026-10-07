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
 <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="18" markerHeight="18" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
  <path d="M0,0 L10,5 L0,10 z" fill="#52514e"/></marker>
 <marker id="ahr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="18" markerHeight="18" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
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



# ------------------------------------------------- simple set (for slides)
def chip(x, y, w, h, fill, label, sub=None, txt="lblw"):
    o = [rect(x, y, w, h, fill, 8),
         text(x + w / 2, y + (h / 2 + 7 if sub is None else h / 2 - 4), label,
              txt, "middle")]
    if sub:
        o.append(text(x + w / 2, y + h / 2 + 22, sub, txt, "middle",
                      'style="font-weight:400;font-size:16px"'))
    return "\n".join(o)


def figA():
    o = [HEAD.format(title="expert 스트리밍 한눈에")]
    o.append(text(80, 92, "11.6 GB짜리 expert를 1.4 GB 작업대로 돌린다", "t1"))
    o.append(text(80, 140, "전부 메모리에 올릴 수 없으니 창고(flash)에 두고, 지금·곧 쓸 것만 작업대(DRAM)에 올려 NPU가 계산한다", "t2"))
    # to-scale size bars
    W = 1760
    o.append(text(80, 214, "expert 전체 3,840개 = 11.6 GB (flash)", "h3"))
    for l in range(30):
        o.append(rect(80 + l * W / 30, 230, W / 30 - 3, 54, "var(--read)", 3, None, 'opacity="0.75"'))
    o.append(text(80, 318, "작업대에 올라가는 양 = 480칸 ≈ 1.4 GB (12%)", "h3"))
    o.append(rect(80, 334, W * 1.4 / 11.6, 54, "var(--npu)", 4))
    o.append(text(80 + W * 1.4 / 11.6 + 16, 370, "← 나머지 88%는 필요할 때 창고에서 가져온다", "lbl"))
    # warehouse -> workbench -> worker
    boxes = [(80, "창고", "Flash (UFS)", ["모든 expert가 있다", "크지만 느리다: 3.0 GB/s"], "var(--read-t)", "var(--read)"),
             (700, "작업대", "DRAM 칸 480개", ["지금 쓰는 것 + 곧 쓸 것", "빈 칸은 다 쓴 것부터 비운다"], "#e6eef9", "var(--npu)"),
             (1320, "작업자", "NPU (Hexagon)", ["작업대 위의 expert만", "가지고 계산한다"], "var(--npu-t)", "var(--npu)")]
    for x, h, sub, lines, bg, ac in boxes:
        o.append(rect(x, 440, 520, 250, bg, 14))
        o.append(text(x + 32, 500, h, "big", extra='style="font-size:40px"'))
        o.append(text(x + 32 + 40 * len(h) + 18, 498, sub, "lane2", extra='style="font-size:20px"'))
        for i, ln in enumerate(lines):
            o.append(text(x + 32, 560 + 40 * i, ln, "lbl", extra='style="font-size:24px"'))
    o.append('<path d="M604,565 L692,565" stroke="#eb6834" stroke-width="5" marker-end="url(#ahr)"/>')
    o.append(text(650, 540, "가져오기", "sm", "middle"))
    o.append('<path d="M1224,565 L1312,565" stroke="#52514e" stroke-width="5" marker-end="url(#ah)"/>')
    o.append(text(1270, 540, "계산", "sm", "middle"))
    # two situations
    o.append(rect(80, 740, 860, 250, "#fff", 14, None, 'style="stroke:var(--ring);stroke-width:2"'))
    o.append(rect(80, 740, 10, 250, "var(--npu)", 4))
    o.append(text(118, 792, "prefill (프롬프트 읽기)", "h3", extra='style="font-size:26px"'))
    o.append(text(118, 842, "무엇이 필요한지 미리 안다 (층마다 거의 전부)", "lbl", extra='style="font-size:22px"'))
    o.append(text(118, 882, "→ 계산하는 동안 다음 층 것을 미리 가져온다", "lbl", extra='style="font-size:22px;font-weight:700"'))
    o.append(text(118, 930, "기다림 7.10 s → 0.01 s (실측, 칸 5 → 16개/층)", "sm"))
    o.append(rect(980, 740, 860, 250, "#fff", 14, None, 'style="stroke:var(--ring);stroke-width:2"'))
    o.append(rect(980, 740, 10, 250, "var(--read)", 4))
    o.append(text(1018, 792, "decode (한 토큰씩 생성)", "h3", extra='style="font-size:26px"'))
    o.append(text(1018, 842, "라우터가 고르기 전에는 모른다 (128개 중 8개)", "lbl", extra='style="font-size:22px"'))
    o.append(text(1018, 882, "→ 쓴 것을 작업대에 남겨 두고, 없으면 그때 가져온다", "lbl", extra='style="font-size:22px;font-weight:700"'))
    o.append(text(1018, 930, "작업대에 있을 확률(히트율)이 속도를 정한다", "sm"))
    o.append(text(80, 1040, "nntrainer의 일반 FSU(층 단위 swap)가 아니라 MoE 층이 expert 단위로 관리하는 경로. "
                            "크기는 문서 55 §3.2 산술, 기다림은 55 §10.11 실측(MoE만 NPU 구성, 2026-10-02).", "xs"))
    o.append(TAIL)
    return "\n".join(o)


def figB():
    o = [HEAD.format(title="prefill 선읽기 전후")]
    o.append(text(80, 92, "prefill: 계산하는 동안 다음 층을 미리 가져오면 기다림이 사라진다", "t1"))
    o.append(text(80, 140, "같은 프롬프트(446토큰), 같은 모델. 달라진 것은 작업대 크기(층당 칸 수) 하나다", "t2"))
    x0 = 330
    w = 170                          # one layer's compute
    wait = (10.94 / 5.01 - 1) * w    # the two totals keep the measured ratio
    # panel 1: no read-ahead
    o.append(text(80, 222, "칸 5개/층", "lane", extra='style="font-size:26px"'))
    o.append(text(80, 256, "prefill 10.94 s", "h3", extra='fill="#d03b3b"'))
    x = x0
    for i in range(4):
        o.append(chip(x, 200, wait - 4, 72, "var(--crit)", "⚠ 기다림", "flash에서 읽기"))
        x += wait
        o.append(chip(x, 200, w - 4, 72, "var(--npu)", f"층 {i + 1} 계산"))
        x += w
    o.append(text(x0, 300, "읽기와 계산이 번갈아: 층마다 expert를 다 읽을 때까지 NPU가 논다", "sm"))
    # panel 2: read-ahead
    o.append(text(80, 392, "칸 16개/층", "lane", extra='style="font-size:26px"'))
    o.append(text(80, 426, "prefill 5.01 s", "h3", extra='fill="#2a78d6"'))
    for i in range(4):
        o.append(chip(x0 + i * w + 8, 346, w - 20, 64, "var(--read)", f"층 {i + 2}", "가져오기"))
        o.append(chip(x0 + i * w, 418, w - 4, 72, "var(--npu)", f"층 {i + 1} 계산"))
    o.append(text(x0 + 4 * w + 20, 384, "← flash: 다음 층 것", "sm"))
    o.append(text(x0 + 4 * w + 20, 462, "← NPU: 쉬지 않고 계산", "sm"))
    o.append(text(x0, 530, "기다림 칸이 없다: 층 N을 계산하는 동안 층 N+1이 이미 작업대에 올라온다", "lbl",
                  extra='style="font-weight:700"'))
    # why: workbench size
    o.append(text(80, 610, "왜 작업대 크기가 결정적인가", "h3", extra='style="font-size:26px"'))
    def bench(y, total, segs, label, note):
        sc = 1500 / 480
        o.append(text(80, y + 34, label, "lbl", extra='style="font-weight:700"'))
        x = 330
        for n, f, lb in segs:
            o.append(rect(x, y, n * sc - 3, 50, f, 4))
            if lb:
                o.append(text(x + n * sc / 2, y + 32, lb, "lblw" if f != "var(--idle)" else "sm", "middle"))
            x += n * sc
        o.append(text(330 + total * sc + 14, y + 32, note, "sm"))
    bench(630, 150, [(128, "var(--npu)", "지금 층 128"), (22, "var(--idle)", "")],
          "칸 5개/층 = 150칸", "남는 22칸 → 다음 층(128)이 안 들어간다")
    bench(710, 480, [(128, "var(--npu)", "지금 층 128"), (128, "var(--npu-t)", ""),
                     (128, "var(--read)", "그다음 층 가져오는 중"), (96, "var(--idle)", "")],
          "칸 16개/층 = 480칸", "")
    o.append(text(330 + 128 * 1500 / 480 + 200, 742, "다음 층 128 (준비됨)", "lbl", "middle"))
    # cards
    o.append(card(80, 800, 560, 200, "var(--crit)", "칸 5개/층", "기다림 7.10 s",
                  ["flash에서 기다리며 읽은 expert 2,893개"], "⚠ "))
    o.append(card(680, 800, 560, 200, "var(--npu)", "칸 16개/층 (기본)", "기다림 0.01 s",
                  ["미리 가져온 expert 3,360개"]))
    o.append(card(1280, 800, 560, 200, "var(--muted)", "그래도 남는 바닥", "≈ 3.3 s",
                  ["prefill마다 약 10 GB를 3.0 GB/s로 읽는다 (산술)"]))
    o.append(text(80, 1050, "실측: Galaxy S25 Ultra, MoE만 NPU 구성, 2026-10-02 (문서 55 §10.11). 막대는 개념도이고 "
                            "두 줄의 길이 비만 실측(10.94 : 5.01)에 맞췄다. 모든 연산 NPU 구성은 기기 미측정.", "xs"))
    o.append(TAIL)
    return "\n".join(o)


def figC():
    o = [HEAD.format(title="decode 작업대 히트")]
    o.append(text(80, 92, "decode: 작업대에 있으면 바로 계산, 없으면 가져올 때까지 기다린다", "t1"))
    o.append(text(80, 140, "토큰마다 라우터가 128개 중 8개를 고른다. 무엇을 고를지 미리 모르니, 최근에 쓴 것을 작업대에 남겨 둔다", "t2"))
    toks = [("토큰 1", [1, 1, 1, 1, 1, 1, 1, 1]), ("토큰 2", [1, 1, 0, 1, 1, 1, 1, 1]),
            ("토큰 3", [1, 1, 1, 1, 1, 1, 1, 1]), ("토큰 4", [1, 0, 1, 1, 0, 1, 1, 1])]
    x = 80
    for name, hits in toks:
        miss = hits.count(0)
        wait = 120 * miss
        width = 300 + wait
        o.append(text(x, 214, name, "h3"))
        for i, h in enumerate(hits):
            o.append(rect(x + i * 36, 232, 30, 30, "var(--npu)" if h else "var(--read)", 4))
        o.append(text(x, 292, f"8개 중 {8 - miss}개 작업대에 있음" if miss else "8개 모두 작업대에 있음", "sm"))
        cx = x
        if miss:
            o.append(chip(cx, 312, wait - 4, 70, "var(--crit)", "⚠ 대기", f"{miss}개 읽기"))
            cx += wait
        o.append(chip(cx, 312, 296, 70, "var(--npu)", "MoE 계산"))
        x += width + 40
    o.append('<path d="M80,412 L1830,412" stroke="#c3c2b7" stroke-width="2" marker-end="url(#ah)"/>')
    o.append(text(1836, 418, "시간", "sm"))
    for i, (f, lb) in enumerate([("var(--npu)", "작업대에 있음 (히트)"), ("var(--read)", "없음 (미스)")]):
        o.append(rect(80 + i * 300, 442, 22, 22, f, 4))
        o.append(text(112 + i * 300, 460, lb, "sm"))
    # chart: hit rate vs decode TPS (LFM2, measured)
    o.append(text(80, 540, "작업대에 있을 확률이 높을수록 빠르다", "h3", extra='style="font-size:26px"'))
    o.append(text(80, 572, "LFM2-8B-A1B 실측 (층당 expert 32개), decode 속도", "sm"))
    rows = [("칸 8개/층", 57, 17.9), ("칸 16개/층", 85, 21.7), ("전부 올림", 100, 24.0)]
    bx, bw_max = 420, 1100
    for i, (lb, hr, tps) in enumerate(rows):
        y = 600 + i * 92
        o.append(text(80, y + 40, lb, "lbl", extra='style="font-size:22px;font-weight:700"'))
        o.append(text(80, y + 68, f"히트 {hr}%", "sm"))
        bwid = bw_max * tps / 24.0
        o.append(rect(bx, y + 12, bwid, 52, "var(--npu)", 4))
        o.append(text(bx + bwid + 14, y + 48, f"{tps} 토큰/s", "lbl", extra='style="font-size:22px;font-weight:700"'))
    o.append('<path d="M420,598 L420,880" stroke="#c3c2b7" stroke-width="2"/>')
    o.append(rect(80, 900, 1760, 100, "#fff", 12, None, 'style="stroke:var(--ring);stroke-width:2"'))
    o.append(text(110, 942, "Gemma-4 26B-A4B는 층당 expert가 128개라 같은 칸 16개로는 히트율이 더 낮을 수 있다 — 기기 미측정.", "lbl"))
    o.append(text(110, 976, "작업대 정리: 쓴 8개와 라우터가 다음으로 꼽은 5개를 \"최근\"으로 올리고, 가장 오래 안 쓴 칸부터 비운다(LRU).", "sm"))
    o.append(text(80, 1050, "출처: 문서 53 §3(LFM2, 2026-09-29 실측). 토큰별 히트/미스는 예시.", "xs"))
    o.append(TAIL)
    return "\n".join(o)



# --------------------------------------- prefill, presentation style
def figP():
    o = [HEAD.format(title="Prefill Expert Prefetch")]
    o.append(text(80, 88, "Prefill: Expert Prefetch", "t1"))
    o.append(text(80, 134, "Layer N 연산 중 Layer N+2의 expert 128개를 Flash에서 미리 읽어(N+1은 이미 준비됨), Flash I/O를 NPU 연산과 overlap", "t2"))
    x0, c = 360, 280
    r = (8.137 / 6.250 - 1) * c         # totals keep the measured ratio

    def lane_lbl(y, s):
        o.append(text(340, y + 36, s, "sm", "end", 'style="font-weight:500"'))

    def blk(x, y, w, fill, s, size=17):
        o.append(rect(x, y, w - 4, 56, fill, 5))
        o.append(text(x + (w - 4) / 2, y + 35, s, "lblw", "middle", f'style="font-size:{size}px"'))

    def idle(x, y, w):
        o.append(f'<rect x="{x}" y="{y}" width="{w - 4}" height="56" rx="5" fill="none" '
                 f'stroke="#898781" stroke-width="2" stroke-dasharray="7 6"/>')
        o.append(text(x + (w - 4) / 2, y + 35, "idle", "xs", "middle"))

    # baseline
    o.append(text(80, 236, "Prefetch off", "h3"))
    o.append(text(80, 264, "On-demand read", "sm"))
    o.append(text(80, 290, "Expert cache 480 slot (1.45 GB)", "sm"))
    lane_lbl(214, "NPU")
    lane_lbl(280, "Flash")
    for i in range(4):
        t = x0 + i * (r + c)
        idle(t, 214, r)
        blk(t, 280, r, "var(--read)", f"Read L{i + 1}", 15)
        blk(t + r, 214, c, "var(--npu)", f"Compute L{i + 1}")
    # prefetch
    o.append(text(80, 412, "Prefetch on", "h3"))
    o.append(text(80, 440, "Read-ahead, LRU", "sm"))
    o.append(text(80, 466, "Expert cache 480 slot (1.45 GB)", "sm"))
    lane_lbl(390, "NPU")
    lane_lbl(456, "Flash")
    for i in range(4):
        blk(x0 + i * c, 390, c, "var(--npu)", f"Compute L{i + 1}")
        blk(x0 + i * c + 8, 456, 0.77 * c, "var(--read)", f"Prefetch L{i + 3}", 15)
    # time axis and the saving
    xe1, xe2 = x0 + 4 * (r + c), x0 + 4 * c
    o.append(f'<path d="M{x0},540 L{xe1 + 30},540" stroke="#c3c2b7" stroke-width="2" marker-end="url(#ah)"/>')
    o.append(text(xe1 + 36, 546, "time", "xs"))
    o.append(f'<path d="M{xe2 + 14},423 L{xe1 - 6},423" stroke="#52514e" stroke-width="2" '
             f'stroke-dasharray="4 5" marker-end="url(#ah)" marker-start="url(#ah)"/>')
    o.append(text((xe2 + xe1) / 2, 410, "Prefill latency −23%", "lbl", "middle", 'style="font-weight:700"'))
    o.append(text((xe2 + xe1) / 2, 458, "8.14 s → 6.25 s", "sm", "middle"))
    o.append(text(x0, 576, "Gantt는 개념도: 4개 layer만 표시, 두 행의 전체 길이 비는 실측 비(8.14 : 6.25)", "xs"))

    # cache occupancy
    o.append(text(80, 640, "Expert cache 480 slot 점유, prefetch on (DRAM, NPU-mapped · 30 layer 공유 · LRU)", "h3"))
    sc = 1440 / 480

    def occ(y, name, segs, note):
        o.append(text(80, y + 34, name, "lbl", extra='style="font-weight:700"'))
        x = x0
        for n, f, lb, ink in segs:
            o.append(rect(x, y, n * sc - 3, 50, f, 4))
            if lb:
                o.append(text(x + n * sc / 2, y + 32, lb, ink, "middle", 'style="font-size:17px"'))
            x += n * sc
        if note:
            o.append(text(x + 14, y + 32, note, "sm"))
    occ(684, "480 slot · on", [(128, "var(--npu)", "Layer N (in use)", "lblw"),
                                     (128, "var(--npu-t)", "Layer N+1 (ready)", "lbl"),
                                     (128, "var(--read)", "Layer N+2 (prefetching)", "lblw"),
                                     (96, "var(--idle)", "free", "sm")], "")

    # measured table
    ty = 816
    cols = [(80, "구성"), (470, "Prefill latency"), (790, "I/O stall"),
            (1080, "동기 expert read"), (1420, "Prefetch expert")]
    o.append(rect(80, ty, 1760, 52, "var(--idle)", 6))
    for x, h in cols:
        o.append(text(x + 20, ty + 34, h, "lbl", extra='style="font-weight:700"'))
    rows = [("Prefetch off · cache 480 slot", "8.14 s", "2.22 s", "1,024", "0"),
            ("Prefetch on · cache 480 slot", "6.25 s", "0 ms", "0", "3,360")]
    for k, row in enumerate(rows):
        y = ty + 52 + k * 50
        o.append(f'<path d="M80,{y + 50} L1840,{y + 50}" stroke="#e1e0d9" stroke-width="2"/>')
        for (x, _), v in zip(cols, row):
            o.append(text(x + 20, y + 34, v, "lbl", extra='style="font-weight:700"' if k == 1 else ""))
    o.append(text(80, ty + 186, "Prefetch는 뒤 layer의 router 결과를 미리 알 수 없어 그 layer의 expert 128개를 모두 읽는다(3,360개 ≈ 10.1 GB). "
                                "off는 router가 고른 expert 중 cache에 없는 것만 읽는다(1,024개).", "sm"))
    o.append(text(80, ty + 220, "I/O stall: off는 동기 read 시간, on은 prefetch를 기다린 시간(exposed wait).", "sm"))
    o.append(text(80, 1062, "측정: Galaxy S25 Ultra (Hexagon V79), Gemma-4 26B-A4B, 512-token prompt, MoE·FC·dense FFN NPU 구성(attention·lm_head CPU), "
                            "NNTR_MOE_PREFETCH=0 / 기본값, 2026-10-07.", "xs"))
    o.append(TAIL)
    return "\n".join(o)



# --------------------------------------------- expert memory, as tables
def figT():
    o = [HEAD.format(title="Expert memory and prefetch")]
    o.append(text(80, 88, "Gemma-4 26B-A4B: Expert 메모리와 Prefetch (512-token)", "t1"))
    o.append(text(80, 134, "Expert weight는 Flash에 두고, DRAM의 Expert cache(30 layer 공유)에는 일부만 올린다. 1 slot = expert 1개(3.01 MB)", "t2"))

    def table(x, y, widths, header, rows, title, bold_last=False, hl_col=None):
        o.append(text(x, y, title, "h3"))
        y += 18
        tw = sum(widths)
        o.append(rect(x, y, tw, 44, "var(--idle)", 6))
        cx = x
        for w, h in zip(widths, header):
            o.append(text(cx + 14, y + 30, h, "sm", extra='style="font-weight:700;fill:#0b0b0b"'))
            cx += w
        y += 44
        for k, row in enumerate(rows):
            cx = x
            for j, (w, v) in enumerate(zip(widths, row)):
                st = 'style="font-weight:700"' if (bold_last and k == len(rows) - 1) or j == hl_col else ""
                o.append(text(cx + 14, y + 29, v, "sm" if j == 0 else "lbl", extra=st if j else ""))
                cx += w
            y += 42
            o.append(f'<path d="M{x},{y} L{x + tw},{y}" stroke="#e1e0d9" stroke-width="2"/>')
        return y

    lw = [360, 300]
    y = table(80, 200, lw, ["항목", "크기"], [
        ("MoE layer × expert", "30 × 128 = 3,840개"),
        ("Expert 1개 (int4 + scale)", "3.01 MB (2.87 MiB)"),
        ("Layer 1개 (128 expert)", "385 MB"),
        ("Token 1개 (30 layer × top-8)", "240개 = 722 MB"),
        ("Expert 전체", "11.56 GB"),
    ], "Expert 크기", bold_last=True)
    table(80, y + 56, lw, ["구성", "크기"], [
        ("Expert (QS4CX_WH)", "11.55 GB"),
        ("Attention · dense FFN (QS4CX)", "0.82 GB"),
        ("Embedding = lm_head (Q4_0)", "0.42 GB"),
        ("Router 등 FP32", "0.04 GB"),
        ("합계 (실측 12,240 MiB)", "12.83 GB"),
    ], "모델 파일 (Flash)", bold_last=True)

    rw = [330, 215, 300, 215]
    hdr = ["항목", "240 slot", "480 slot (기본)", "720 slot"]
    y = table(780, 200, rw, hdr, [
        ("설정값 moe_cache_experts (×30)", "8", "16", "24"),
        ("Cache 크기 (DRAM)", "0.72 GB", "1.45 GB", "2.17 GB"),
        ("미리 올려 두는 양 (산술)", "0 (112 < 128)", "256개 ≈ 0.77 GB", "512개 ≈ 1.54 GB"),
        ("Flash read / prefill (상한)", "10.84 GB", "10.12 GB", "9.39 GB"),
        ("Flash 하한 (÷ 3.0 GB/s)", "3.61 s", "3.37 s", "3.13 s"),
        ("Prefetch (실측)", "—", "3,360개 · 대기 0 ms", "—"),
        ("Prefill latency (실측)", "4.85 s", "4.27 s", "4.15 s"),
    ], "Prefill (512-token prompt)", hl_col=2)
    table(780, y + 56, rw, hdr, [
        ("Decode 속도 (실측)", "4.51 TPS", "3.39 TPS", "3.47 TPS"),
        ("Cache miss (실측)", "81,348", "62,198", "41,955"),
        ("Hit rate (계산)", "34%", "49%", "66%"),
        ("Flash read / token", "478 MB", "366 MB", "247 MB"),
        ("Read 시간 / miss (실측)", "0.76 ms", "1.58 ms", "2.27 ms"),
    ], "Decode (512 token 생성)", hl_col=2)
    o.append(text(80, 990, "미리 올려 두는 양: 다음 layer(128개) 단위로만 queue → (slot − 현재 layer 128) ÷ 128 layer. "
                           "Hit rate = 1 − miss ÷ (512 token × 240). Flash read / token = miss ÷ 512 × 3.01 MB.", "xs"))
    o.append(text(80, 1018, "Cache가 커지면 hit는 오르지만 miss당 read가 느려진다(ION arena가 커질수록 OS page cache가 줄어드는 것으로 추정, 미측정).", "xs"))
    o.append(text(80, 1046, "실측: Galaxy S25 Ultra (Hexagon V79), FC·MoE NPU 구성, 2026-10-02 (문서 55 §10.14·10.15, 57 §3). "
                            "이번 브랜치(전 연산 NPU)는 기기 미측정.", "xs"))
    o.append(TAIL)
    return "\n".join(o)


for name, fn in [("fsu_1_placement", fig1), ("fsu_2_prefill_prefetch", fig2),
                 ("fsu_3_decode_cache", fig3), ("fsu_a_overview", figA),
                 ("fsu_b_prefill_before_after", figB), ("fsu_c_decode_hits", figC),
                 ("fsu_prefill_prefetch", figP), ("fsu_expert_memory_table", figT)]:
    open(f"{name}.html", "w").write(fn())
print("ok")
