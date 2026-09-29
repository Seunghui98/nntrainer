#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
@file   attn_fma_cases.py
@date   29 Sep 2026
@brief  Writes the fp16 FMA case file HvxAttnM1Probe.Semantics reads
        (plan 170 step 1): the (c, a, b) triples of a replay of #136's
        dump_attn through attn_m1_det.h's fp16 attention at pos 512
@author dlwlzzero <dlwlzzero@gmail.com>

The fused fp16 FMA (attn_m1_det_fma16, the CPU's fmla .8h) differs from
rounding c + a*b to f32 first only on rare triples: 2.6e-5 per op on real
data (plan 170 section 0). The probe needs those triples, so this replays
the six attention layers of the dump at pos 512 (the kvseed_<layer>_{k,v}
rows plus the dumped row, L = 513) through the spec's score chains
(acc[d % 8], d ascending), the spec's softmax and the PV chains, and keeps
every triple that is a double-rounding hazard or whose f32 sum is an
inexact fp16 midpoint, plus every 3000th op.

The q / k / v rows are the ROPE-ATTN_M1 stretch's input as dumped, i.e.
before RoPE: real magnitudes and mantissas, not the model's exact
attention (the triples, not the outputs, are what the probe needs).

Usage: attn_fma_cases.py <dump_attn dir> <out.bin>
Output: little-endian uint16 fp16 bits, (c, a, b) per triple.
"""

import os
import sys

import numpy as np

NKV, GQA, HD = 8, 4, 64
F16, F32, F64 = np.float16, np.float32, np.float64


def rne16(x):
    """f32 -> the fp16 grid (numpy's conversion rounds once, ties to even)."""
    return x.astype(F16).astype(F32)


class Replay:
    """The spec's fused fp16 FMA with a record of every triple."""

    def __init__(self):
        self.c, self.a, self.b, self.keep = [], [], [], []
        self.ops = self.hazards = self.inexact_mid = 0

    def fma(self, c, a, b):
        c, a, b = np.broadcast_arrays(F32(c), F32(a), F32(b))
        p = a.astype(F64) * b.astype(F64)  # exact: 22 bits
        s = c.astype(F64) + p
        bv = s - c
        err = (c - (s - bv)) + (p - bv)  # TwoSum: s + err is exact
        bits = s.view(np.uint64).copy()
        odd = err != 0  # round to odd at 53 bits, then one conversion
        down = odd & (np.signbit(err) != np.signbit(s))
        bits[down] -= np.uint64(1)
        bits[odd] |= np.uint64(1)
        fused = bits.view(F64).astype(F16).astype(F32)
        s32 = (c + p.astype(F32)).astype(F32)  # the f32 sum, rounded once
        twice = rne16(s32)
        u = s32.view(np.uint32)
        mag = np.abs(s32)
        mid = np.where(mag >= F32(2.0**-14), (u & 0x1FFF) == 0x1000,
                       np.fmod(mag.astype(F64) * 2.0**25, 2.0) == 1.0)
        inexact = (s32.astype(F64) != s) | odd
        hz = fused != twice
        im = mid & inexact
        idx = self.ops + np.arange(c.size)
        self.keep.append((hz | im | (idx % 3000 == 0)).ravel())
        for lst, v in ((self.c, c), (self.a, a), (self.b, b)):
            lst.append(v.astype(F16).ravel())
        self.ops += c.size
        self.hazards += int(hz.sum())
        self.inexact_mid += int(im.sum())
        return fused


def exp16(d):
    """attn_m1_det_exp16 (neon_mathfun's exp_ps, fused fx) on f32 arrays."""
    x = np.minimum(d.astype(F32), F32(88.3762626647949))
    x = np.maximum(x, F32(-88.3762626647949))
    fx = (x.astype(F64) * F64(F32(1.44269504088896341)) + 0.5).astype(F32)
    tmp = np.trunc(fx).astype(F32)
    fx = (tmp - np.where(tmp > fx, F32(1), F32(0))).astype(F32)
    x = (x - fx * F32(0.693359375)).astype(F32)
    x = (x - fx * F32(-2.12194440e-4)).astype(F32)
    p = [F32(v) for v in (1.9875691500E-4, 1.3981999507E-3, 8.3334519073E-3,
                          4.1665795894E-2, 1.6666665459E-1, 5.0000001201E-1)]
    y = p[0] * x
    z = x * x
    for c in p[1:]:
        y = (y + c) * x if c is not p[5] else y + c
    y = y * z + x + F32(1)
    mm = ((fx.astype(np.int32) + 0x7F) << 23).astype(np.uint32).view(F32)
    return rne16((y * mm).astype(F32))


def head(rep, q, K, V):
    """Steps 2-7 of attn_m1_det.h for one q head; K, V: [L][64] fp16."""
    L = K.shape[0]
    acc = np.zeros((8, L), F32)
    for k in range(8):
        for l in range(8):
            d = 8 * k + l
            acc[l] = rep.fma(acc[l], q[d], K[:, d])
    t = [rne16(acc[i] + acc[i + 1]) for i in (0, 2, 4, 6)]
    t = rne16(rne16(t[0] + t[1]) + rne16(t[2] + t[3]))
    s = rne16((F32(0) + t) * F32(0.125))
    m = s.max() + F32(0)
    e = exp16(rne16(s - m))
    l = F32(0)
    for v in e:
        l = rne16(np.array([l + v], F32))[0]
    e = rne16(e / l)
    o = np.zeros(HD, F32)
    for p in range(L):
        o = rep.fma(o, e[p], V[p])


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, out = sys.argv[1], sys.argv[2]
    calls = []
    with open(os.path.join(src, 'forward_manifest.txt')) as f:
        for line in f:
            w = line.split()
            if w and w[-1] == 'ROPE-ATTN_M1' and w[3] == '512':
                calls.append(w[0])
    if len(calls) != 6:
        sys.exit('expected 6 ROPE-ATTN_M1 calls at pos 512, found %d' %
                 len(calls))
    rep = Replay()
    for layer, name in enumerate(calls):
        rd = lambda n: np.fromfile(os.path.join(src, n), F32)
        row = rd(name + '_in.f32')
        K = np.vstack([rd('kvseed_%d_k.f32' % layer).reshape(512, NKV * HD),
                       row[2048:2560]])
        V = np.vstack([rd('kvseed_%d_v.f32' % layer).reshape(512, NKV * HD),
                       row[2560:3072]])
        K, V = rne16(K), rne16(V)
        for hq in range(NKV * GQA):
            h = hq // GQA
            head(rep, rne16(row[hq * HD:(hq + 1) * HD]),
                 K[:, h * HD:(h + 1) * HD], V[:, h * HD:(h + 1) * HD])
    keep = np.concatenate(rep.keep)
    trip = np.stack([np.concatenate(x)[keep] for x in (rep.c, rep.a, rep.b)],
                    axis=1)
    trip.view(np.uint16).astype('<u2').tofile(out)
    print('attn_fma_cases: ops=%d hazards=%d inexact_midpoints=%d '
          'written=%d -> %s' % (rep.ops, rep.hazards, rep.inexact_mid,
                                trip.shape[0], out))


if __name__ == '__main__':
    main()
