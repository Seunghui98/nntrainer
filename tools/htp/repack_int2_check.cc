// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   repack_int2_check.cc
 * @date   07 Oct 2026
 * @brief  Checks a repack_int2_to_qs2cx.py output against its input, every
 *         byte, through the loader's own QS2CX_WH reader (whUnpack2)
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 *   repack_int2_check <in.bin> <out.bin> <table>
 *
 * The table is the script's: one line per segment, `C in_off out_off len 0`
 * (copied) or `E in_off out_off K N` (an expert). The segments must tile
 * both files with no gap. Per expert: the input's codes decoded by the
 * external layout (tile byte sl/4, shift 2*(sl%4), value code - 2) must
 * sum, per column, to the input's colsum -- the check that the layout
 * assumption still holds for the file at hand -- and equal whUnpack2 of the
 * output's codes element for element; the output's palette is fe ff 00 01;
 * scale and colsum are copied. Prints `REPACK CHECK ... ok` or exits 1.
 * Built and run by the script (--verify-only to rerun it alone).
 */

#include "htp_wh_palette.h"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace nntrainer;

namespace {

struct Seg {
  char kind;
  size_t in_off, out_off, a, b;
};

const uint8_t *mapFile(const char *path, size_t *size) {
  const int fd = open(path, O_RDONLY);
  struct stat st;
  if (fd < 0 || fstat(fd, &st) != 0) {
    std::perror(path);
    return nullptr;
  }
  *size = static_cast<size_t>(st.st_size);
  void *p = mmap(nullptr, *size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  return p == MAP_FAILED ? nullptr : static_cast<const uint8_t *>(p);
}

/** @return 0 if the expert checks, else which check failed (1..4) */
int checkExpert(const uint8_t *in, const uint8_t *out, uint32_t K, uint32_t N,
                std::vector<int8_t> &want, std::vector<int8_t> &got) {
  const size_t codes = static_cast<size_t>(K) * N / 4;
  const uint32_t n_tiles = N / WH_TILE;
  want.assign(static_cast<size_t>(K) * N, 0);
  got.assign(want.size(), 0);
  for (uint32_t kt = 0; kt < K / WH_TILE; ++kt)
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
      const uint8_t *tile = in + ((size_t)kt * n_tiles + nt) * WH_TILE2_BYTES;
      for (uint32_t r = 0; r < WH_TILE; ++r)
        for (uint32_t c = 0; c < WH_TILE; ++c) {
          const uint32_t sl = whSlot(r, c);
          want[(size_t)(kt * WH_TILE + r) * N + nt * WH_TILE + c] =
            static_cast<int8_t>(((tile[sl / 4] >> (2 * (sl % 4))) & 3) - 2);
        }
    }
  std::vector<float> colsum(N);
  std::memcpy(colsum.data(), in + codes + 4 * (size_t)N, 4 * (size_t)N);
  for (uint32_t n = 0; n < N; ++n) {
    long s = 0;
    for (uint32_t k = 0; k < K; ++k)
      s += want[(size_t)k * N + n];
    if (static_cast<float>(s) != colsum[n])
      return 1;
  }
  static const uint8_t kPalette[4] = {0xfe, 0xff, 0x00, 0x01};
  if (std::memcmp(out + codes, kPalette, 4) != 0)
    return 2;
  whUnpack2(out, K, N, reinterpret_cast<const int8_t *>(out + codes),
            got.data());
  if (want != got)
    return 3;
  if (std::memcmp(in + codes, out + codes + 4, 8 * (size_t)N) != 0)
    return 4;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <in.bin> <out.bin> <table>\n", argv[0]);
    return 2;
  }
  size_t in_size = 0, out_size = 0;
  const uint8_t *in = mapFile(argv[1], &in_size);
  const uint8_t *out = mapFile(argv[2], &out_size);
  FILE *tf = std::fopen(argv[3], "r");
  if (in == nullptr || out == nullptr || tf == nullptr)
    return 1;
  std::vector<Seg> segs;
  Seg s;
  size_t ic = 0, oc = 0;
  while (std::fscanf(tf, " %c %zu %zu %zu %zu", &s.kind, &s.in_off, &s.out_off,
                     &s.a, &s.b) == 5) {
    if (s.in_off != ic || s.out_off != oc) {
      std::printf("REPACK CHECK FAIL gap at segment %zu\n", segs.size());
      return 1;
    }
    const size_t n = s.kind == 'C' ? s.a : s.a * s.b / 4 + 8 * s.b;
    ic += n;
    oc += n + (s.kind == 'E' ? 4 : 0);
    segs.push_back(s);
  }
  std::fclose(tf);
  if (ic != in_size || oc != out_size) {
    std::printf("REPACK CHECK FAIL coverage in %zu/%zu out %zu/%zu\n", ic,
                in_size, oc, out_size);
    return 1;
  }

  size_t experts = 0, copies = 0, elements = 0, bad = 0;
#pragma omp parallel reduction(+ : experts, copies, elements, bad)
  {
    std::vector<int8_t> want, got;
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < segs.size(); ++i) {
      const Seg &g = segs[i];
      int err = 0;
      if (g.kind == 'C') {
        err = std::memcmp(in + g.in_off, out + g.out_off, g.a) != 0 ? 5 : 0;
        ++copies;
      } else {
        err = checkExpert(in + g.in_off, out + g.out_off, g.a, g.b, want, got);
        ++experts;
        elements += g.a * g.b;
      }
      if (err) {
        ++bad;
#pragma omp critical
        std::printf("REPACK CHECK FAIL segment %zu %c in_off=%zu check=%d\n", i,
                    g.kind, g.in_off, err);
      }
    }
  }
  std::printf("REPACK CHECK experts=%zu copies=%zu elements=%zu "
              "mismatches=%zu %s\n",
              experts, copies, elements, bad, bad ? "FAIL" : "ok");
  return bad ? 1 : 0;
}
