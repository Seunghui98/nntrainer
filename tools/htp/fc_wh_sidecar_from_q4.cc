// SPDX-License-Identifier: Apache-2.0
/**
 * @file   fc_wh_sidecar_from_q4.cc
 * @brief  fc_wh_sidecar_from_q4.py's worker: the FC WH sidecar of a main
 *         file whose FCs are ARM Q4_0 (q4_0x4), each image re-quantized by
 *         htp_qs4cx_from_q4_0x4 and packed by whPack
 * @author dlwlzzero <dlwlzzero@gmail.com>
 *
 * fcwh <main.bin> <table> <out.bin> [<check sidecar>]
 * table: one "name offset K N" line per FC weight, in file order.
 * With a check sidecar: its index must equal ours byte for byte (the same
 * names, shapes, keys, offsets and lengths), and each image's weight
 * (code x scale) is compared with ours by SNR, printed per weight and as a
 * minimum. Exit 0 on success.
 */
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <htp_q4_0_convert.h>
#include <htp_wh_layout.h>

namespace {

struct Row {
  std::string name;
  uint64_t off;
  uint32_t K, N;
};

/** @brief code x scale of one image, K x N row-major. */
std::vector<float> weight(const std::vector<uint8_t> &img, uint32_t K,
                          uint32_t N) {
  std::vector<int8_t> rm(static_cast<size_t>(K) * N);
  nntrainer::whUnpack(img.data(), K, N, rm.data());
  const float *s =
    reinterpret_cast<const float *>(img.data() + nntrainer::whBytes(K, N));
  std::vector<float> w(rm.size());
  for (size_t k = 0; k < K; ++k)
    for (size_t n = 0; n < N; ++n)
      w[k * N + n] = rm[k * N + n] * s[n];
  return w;
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: %s <main.bin> <table> <out.bin> [<check>]\n",
                 argv[0]);
    return 2;
  }
  std::ifstream main_f(argv[1], std::ios::binary), tab(argv[2]);
  std::vector<Row> rows;
  for (Row r; tab >> r.name >> r.off >> r.K >> r.N;)
    rows.push_back(r);
  if (!main_f || rows.empty()) {
    std::fprintf(stderr, "cannot read %s, or no row in %s\n", argv[1], argv[2]);
    return 1;
  }
  const uint32_t count = static_cast<uint32_t>(rows.size());
  std::vector<nntrainer::FcWhEntry> idx(count);
  std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
  const uint64_t head = nntrainer::fcWhHeaderBytes(count);
  out.write(std::string(head, '\0').data(), static_cast<std::streamsize>(head));
  std::set<uint64_t> keys;
  uint64_t at = head;
  for (uint32_t i = 0; i < count; ++i) {
    const Row &r = rows[i];
    if (r.K % 32u || r.N % 32u || r.name.size() >= sizeof(idx[i].name)) {
      std::fprintf(stderr,
                   "%s: %u x %u is not whole WH tiles, or the name "
                   "is too long\n",
                   r.name.c_str(), r.K, r.N);
      return 1;
    }
    std::vector<char> q(static_cast<size_t>(r.N) * (r.K / 32u) * 18u);
    main_f.seekg(static_cast<std::streamoff>(r.off));
    main_f.read(q.data(), static_cast<std::streamsize>(q.size()));
    if (!main_f) {
      std::fprintf(stderr, "%s: cannot read %zu B at %llu\n", r.name.c_str(),
                   q.size(), static_cast<unsigned long long>(r.off));
      return 1;
    }
    std::vector<int8_t> rm(static_cast<size_t>(r.K) * r.N);
    std::vector<float> scale(r.N);
    std::vector<int32_t> colsum(r.N);
    nntrainer::htp_qs4cx_from_q4_0x4(q.data(), r.K, r.N, rm.data(),
                                     scale.data(), colsum.data());
    std::vector<uint8_t> img(nntrainer::whBytes(r.K, r.N));
    nntrainer::whPack(rm.data(), r.K, r.N, img.data());
    std::vector<float> cs(colsum.begin(), colsum.end());
    out.write(reinterpret_cast<const char *>(img.data()),
              static_cast<std::streamsize>(img.size()));
    out.write(reinterpret_cast<const char *>(scale.data()),
              static_cast<std::streamsize>(4u * r.N));
    out.write(reinterpret_cast<const char *>(cs.data()),
              static_cast<std::streamsize>(4u * r.N));
    nntrainer::FcWhEntry &e = idx[i];
    std::memcpy(e.name, r.name.data(), r.name.size());
    e.K = r.K;
    e.N = r.N;
    e.key = nntrainer::fcWhKey(q.data(), q.size());
    e.q4_off = r.off;
    e.off = at;
    e.bytes = img.size() + 8u * r.N;
    at += e.bytes;
    if (!keys.insert(e.key).second) {
      // set_fc_wh_file refuses a duplicate key at load: say so here
      std::fprintf(stderr, "%s: fcWhKey collides with an earlier weight\n",
                   r.name.c_str());
      return 1;
    }
  }
  const uint32_t version = nntrainer::FCWH_VERSION;
  out.seekp(0);
  out.write(nntrainer::FCWH_MAGIC, sizeof(nntrainer::FCWH_MAGIC));
  out.write(reinterpret_cast<const char *>(&version), 4);
  out.write(reinterpret_cast<const char *>(&count), 4);
  out.write(reinterpret_cast<const char *>(idx.data()),
            static_cast<std::streamsize>(count * sizeof(nntrainer::FcWhEntry)));
  out.close();
  if (!out) {
    std::fprintf(stderr, "cannot write %s\n", argv[3]);
    return 1;
  }
  std::printf("fc wh sidecar: %s images=%u bytes=%llu keys_unique=1\n"
              "nntr_config.json: \"fc_wh_file_name\": \"<its file name>\", "
              "\"fc_wh_format\": \"%s\"\n",
              argv[3], count, static_cast<unsigned long long>(at),
              nntrainer::FCWH_FORMAT);
  if (argc == 4)
    return 0;

  // The check: the index as quantize_stream wrote it, the images by SNR.
  std::ifstream ours(argv[3], std::ios::binary), ref(argv[4], std::ios::binary);
  std::vector<char> ha(head), hb(head);
  ours.read(ha.data(), static_cast<std::streamsize>(head));
  ref.read(hb.data(), static_cast<std::streamsize>(head));
  const bool same_index = ref && ha == hb;
  double min_snr = INFINITY;
  for (const auto &e : idx) {
    std::vector<uint8_t> a(e.bytes), b(e.bytes);
    ours.seekg(static_cast<std::streamoff>(e.off));
    ref.seekg(static_cast<std::streamoff>(e.off));
    ours.read(reinterpret_cast<char *>(a.data()),
              static_cast<std::streamsize>(e.bytes));
    ref.read(reinterpret_cast<char *>(b.data()),
             static_cast<std::streamsize>(e.bytes));
    if (!ours || !ref) {
      std::fprintf(stderr, "check: cannot read %s\n", e.name);
      return 1;
    }
    const std::vector<float> wa = weight(a, e.K, e.N), wb = weight(b, e.K, e.N);
    double sig = 0.0, err = 0.0;
    for (size_t i = 0; i < wa.size(); ++i) {
      sig += static_cast<double>(wb[i]) * wb[i];
      err += static_cast<double>(wa[i] - wb[i]) * (wa[i] - wb[i]);
    }
    const double snr = err == 0.0 ? INFINITY : 10.0 * std::log10(sig / err);
    min_snr = snr < min_snr ? snr : min_snr;
    std::printf("check %s %ux%u snr_db=%.2f\n", e.name, e.K, e.N, snr);
  }
  std::printf("FC WH FROM Q4 CHECK index_identical=%d images=%u "
              "min_snr_db=%.2f\n",
              same_index ? 1 : 0, count, min_snr);
  return same_index ? 0 : 1;
}
