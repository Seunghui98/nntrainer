#!/usr/bin/env python3
##
# @file    276-qs2cx-check.py
# @brief   #276 S1: the 2-bit Gemma-4 file's layout (offsets from the
#          writer's order) and a palette / code / colsum check of expert
#          tensors read back from the device
#
# layout <config.json> <fc dtype Q4_0|QS4CX> [name...]
#     Walks writeGemma4Moe (quantize_stream.cpp @ c60bb34df, the producer's
#     commit): Q4_0 embedding, per layer the norms / Q4_0 FCs / router /
#     128 x (gate_up, down) QS2CX_WH / tail norms, output_norm and, untied,
#     the Q4_0 head. Prints "total <bytes>" (must equal the file's size) and
#     "<name> <offset> <size>" for each name asked.
# check <tensors.txt> <dir>
#     For each "<name> <offset> <size>" line, <dir>/<name>.raw holds the
#     tensor's bytes: [K*N/4 codes in whPack2 order][4 palette int8][N f32
#     scale][N f32 colsum]. Prints the palette, the 2-bit code histogram,
#     the decoded int4 value histogram, scale min / max, and whether every
#     column's decoded sum equals the stored colsum (the writer computes it
#     from the same values, quantize_stream.cpp:1075-1089).
#     The decode follows whUnpack2 (htp_wh_palette.h:291): whSlot(r, c),
#     whCodeByte2 / whCodeShift2 (hvx_expand_i2i4.h:90-99).
import json
import sys

import numpy as np

TILE, GROUP = 32, 128


def layout(cfg_f, names, fc="Q4_0"):
    c = json.load(open(cfg_f))
    H, V, L, E = (c["hidden_size"], c["vocab_size"], c["num_hidden_layers"],
                  c["num_experts"])
    I, MI = c["intermediate_size"], c["moe_intermediate_size"]
    nh, hd, ghd = c["num_attention_heads"], c["head_dim"], c["global_head_dim"]
    kvh, gkvh = c["num_key_value_heads"], c["num_global_key_value_heads"]
    keqv = c["attention_k_eq_v"]
    q40 = lambda k, n: k * n // 32 * 18
    # the FCs (not the embedding / head): Q4_0, or QS4CX = K*N/2 codes + N f32
    fcb = q40 if fc == "Q4_0" else (lambda k, n: k * n // 2 + 4 * n)
    f32 = lambda n: 4 * n
    qs2 = lambda k, n: k * n // 4 + 4 + 8 * n
    t, off = [], 0

    def add(name, size, k=0, n=0):
        nonlocal off
        t.append((name, off, size, k, n))
        off += size

    add("embedding0", q40(V, H))
    for li in range(L):
        p = "layer%d" % li
        sl = c["layer_types"][li] == "sliding_attention"
        d = hd if sl else ghd
        kv = kvh if (sl or not keqv) else gkvh
        add(p + "_attention_norm", f32(H))
        add(p + "_wq", fcb(H, nh * d))
        add(p + "_q_norm", f32(d))
        add(p + "_wk", fcb(H, kv * d))
        add(p + "_k_norm", f32(d))
        if not keqv or sl:
            add(p + "_wv", fcb(H, kv * d))
        add(p + "_attention_out", fcb(nh * d, H))
        for n in ("_post_attention_norm", "_pre_ffn_norm"):
            add(p + n, f32(H))
        add(p + "_ffn_gate", fcb(H, I))
        add(p + "_ffn_up", fcb(H, I))
        add(p + "_ffn_down", fcb(I, H))
        for n in ("_post_ffn_norm_1", "_pre_ffn_norm_2", "_router_norm"):
            add(p + n, f32(H))
        add(p + "_router", f32(H * E))
        add(p + "_per_expert_scale", f32(E))
        for e in range(E):
            add("%s_expert%d_gate_up" % (p, e), qs2(H, 2 * MI), H, 2 * MI)
            add("%s_expert%d_down" % (p, e), qs2(MI, H), MI, H)
        for n in ("_post_ffn_norm_2", "_post_ffn_norm"):
            add(p + n, f32(H))
        add(p + "_layer_scalar", f32(1))
    add("output_norm", f32(H))
    if not c.get("tie_word_embeddings", True):
        add("output_of_causallm", q40(H, V))
    print("total", off)
    for name, o, s, k, n in t:
        if name in names:
            print(name, o, s, k, n)


def unpack2(codes, K, N):
    """Code index (0..3) per [k][n], whUnpack2's order."""
    r = np.arange(TILE)[:, None]
    cc = np.arange(TILE)[None, :]
    sl = (r // 8) * 256 + cc * 8 + (r % 4) * 2 + ((r // 4) % 2)
    ob = sl >> 1
    byte = (ob >> 8) * GROUP + (ob & (GROUP - 1))
    shift = 2 * (sl & 1) + np.where((ob >> 7) & 1, 4, 0)
    kt, nt = K // TILE, N // TILE
    tiles = codes.reshape(kt, nt, TILE * TILE // 4)
    idx = (tiles[:, :, byte] >> shift) & 3  # [kt, nt, r, c]
    return idx.transpose(0, 2, 1, 3).reshape(K, N)


def check(lst, d):
    ok = True
    for line in open(lst):
        name, off, size, K, N = line.split()
        size, K, N = int(size), int(K), int(N)
        raw = np.fromfile("%s/%s.raw" % (d, name), dtype=np.uint8)
        nc = K * N // 4
        if raw.size != size or size != nc + 4 + 8 * N:
            print("CHECK FAIL %s: read %d B, want %d" % (name, raw.size, size))
            ok = False
            continue
        pal = raw[nc:nc + 4].view(np.int8)
        scale = raw[nc + 4:nc + 4 + 4 * N].view(np.float32)
        colsum = raw[nc + 4 + 4 * N:].view(np.float32)
        idx = unpack2(raw[:nc], K, N)
        vals = pal[idx].astype(np.int64)
        good = np.array_equal(vals.sum(axis=0).astype(np.float32), colsum)
        ok &= bool(good)
        hc = np.bincount(idx.ravel(), minlength=4)
        uv, cv = np.unique(vals, return_counts=True)
        print("%s K=%d N=%d palette=%s codes=%s values=%s scale=[%.3g, %.3g] "
              "colsum_match=%d" % (name, K, N, pal.tolist(), hc.tolist(),
                                   dict(zip(uv.tolist(), cv.tolist())),
                                   scale.min(), scale.max(), good))
    print("QS2CX CHECK %s" % ("ok" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    if sys.argv[1] == "layout":
        layout(sys.argv[2], set(sys.argv[4:]), sys.argv[3])
    elif sys.argv[1] == "check":
        sys.exit(check(sys.argv[2], sys.argv[3]))
    else:
        sys.exit(__doc__)
