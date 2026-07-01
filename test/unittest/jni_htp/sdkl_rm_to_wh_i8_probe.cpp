// Minimal probe for sdkl_cpu_rm_to_wh_i8_inplace().
//
// Usage examples:
//   sdkl_rm_to_wh_i8_probe --preset small --buf all
//   sdkl_rm_to_wh_i8_probe --preset qwen_attn --buf host_aligned64
//   sdkl_rm_to_wh_i8_probe --n 3072 --k 1024 --buf npu
//   sdkl_rm_to_wh_i8_probe --preset qwen_ffn_up --buf all --no-init
//
// The goal is to isolate which shape / buffer-kind combination triggers a
// device-side fault during RM -> WH layout conversion for INT8 weights.

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <remote.h>
#include <sdkl.h>

namespace {

enum class BufferKind {
  HOST_UNALIGNED,
  HOST_ALIGNED64,
  NPU,
};

struct Options {
  size_t N = 64;
  size_t K = 64;
  bool init_npu = true;
  std::vector<BufferKind> cases = {BufferKind::HOST_UNALIGNED,
                                   BufferKind::HOST_ALIGNED64, BufferKind::NPU};
};

struct Buffer {
  void *base = nullptr;
  int8_t *data = nullptr;
  size_t size = 0;
  BufferKind kind = BufferKind::HOST_UNALIGNED;
};

const char *toString(BufferKind kind) {
  switch (kind) {
  case BufferKind::HOST_UNALIGNED:
    return "host_unaligned";
  case BufferKind::HOST_ALIGNED64:
    return "host_aligned64";
  case BufferKind::NPU:
    return "npu";
  }

  return "unknown";
}

void printUsage(const char *argv0) {
  std::fprintf(stderr,
               "Usage: %s [--preset NAME] [--n N] [--k K] [--buf KIND] "
               "[--no-init]\n"
               "\n"
               "Presets:\n"
               "  small         N=64   K=64\n"
               "  qwen_attn     N=1024 K=1024\n"
               "  qwen_ffn_up   N=3072 K=1024\n"
               "  qwen_ffn_down N=1024 K=3072\n"
               "\n"
               "Buffer kinds:\n"
               "  host_unaligned\n"
               "  host_aligned64\n"
               "  npu\n"
               "  all\n",
               argv0);
}

bool parsePreset(const char *name, Options &opt) {
  if (std::strcmp(name, "small") == 0) {
    opt.N = 64;
    opt.K = 64;
    return true;
  }

  if (std::strcmp(name, "qwen_attn") == 0) {
    opt.N = 1024;
    opt.K = 1024;
    return true;
  }

  if (std::strcmp(name, "qwen_ffn_up") == 0) {
    opt.N = 3072;
    opt.K = 1024;
    return true;
  }

  if (std::strcmp(name, "qwen_ffn_down") == 0) {
    opt.N = 1024;
    opt.K = 3072;
    return true;
  }

  return false;
}

bool parseBufferKind(const char *name, std::vector<BufferKind> &out) {
  if (std::strcmp(name, "all") == 0) {
    out = {BufferKind::HOST_UNALIGNED, BufferKind::HOST_ALIGNED64,
           BufferKind::NPU};
    return true;
  }

  if (std::strcmp(name, "host_unaligned") == 0) {
    out = {BufferKind::HOST_UNALIGNED};
    return true;
  }

  if (std::strcmp(name, "host_aligned64") == 0) {
    out = {BufferKind::HOST_ALIGNED64};
    return true;
  }

  if (std::strcmp(name, "npu") == 0) {
    out = {BufferKind::NPU};
    return true;
  }

  return false;
}

bool parseSize(const char *s, size_t &out) {
  char *end = nullptr;
  const unsigned long long value = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0' || value == 0)
    return false;

  out = static_cast<size_t>(value);
  return true;
}

bool parseArgs(int argc, char **argv, Options &opt) {
  for (int i = 1; i < argc; ++i) {
    const char *arg = argv[i];

    if (std::strcmp(arg, "--preset") == 0) {
      if (i + 1 >= argc || !parsePreset(argv[++i], opt))
        return false;
      continue;
    }

    if (std::strcmp(arg, "--n") == 0) {
      if (i + 1 >= argc || !parseSize(argv[++i], opt.N))
        return false;
      continue;
    }

    if (std::strcmp(arg, "--k") == 0) {
      if (i + 1 >= argc || !parseSize(argv[++i], opt.K))
        return false;
      continue;
    }

    if (std::strcmp(arg, "--buf") == 0) {
      if (i + 1 >= argc || !parseBufferKind(argv[++i], opt.cases))
        return false;
      continue;
    }

    if (std::strcmp(arg, "--no-init") == 0) {
      opt.init_npu = false;
      continue;
    }

    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      printUsage(argv[0]);
      std::exit(0);
    }

    return false;
  }

  return true;
}

std::vector<int8_t> makeWeights(size_t N, size_t K) {
  std::vector<int8_t> weights(N * K);
  for (size_t n = 0; n < N; ++n) {
    for (size_t k = 0; k < K; ++k) {
      const int value =
        static_cast<int>(((n * 17u + k * 13u + 5u) % 255u)) - 127;
      weights[n * K + k] = static_cast<int8_t>(value);
    }
  }
  return weights;
}

uint64_t fnv1a64(const int8_t *data, size_t bytes) {
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= static_cast<uint8_t>(data[i]);
    hash *= 1099511628211ull;
  }
  return hash;
}

void dumpPrefix(const int8_t *data, size_t bytes, size_t max_dump = 16) {
  const size_t dump = bytes < max_dump ? bytes : max_dump;
  std::fprintf(stderr, "  prefix:");
  for (size_t i = 0; i < dump; ++i)
    std::fprintf(stderr, " %02x", static_cast<uint8_t>(data[i]));
  std::fprintf(stderr, "\n");
}

bool allocateBuffer(BufferKind kind, size_t bytes, Buffer &buf) {
  buf.kind = kind;
  buf.size = bytes;

  if (kind == BufferKind::HOST_UNALIGNED) {
    buf.base = std::malloc(bytes + 64);
    if (buf.base == nullptr)
      return false;

    auto *raw = static_cast<int8_t *>(buf.base);
    buf.data = raw;
    if ((reinterpret_cast<uintptr_t>(buf.data) % 64u) == 0u)
      buf.data += 1;
    return true;
  }

  if (kind == BufferKind::HOST_ALIGNED64) {
    void *ptr = nullptr;
    if (posix_memalign(&ptr, 64, bytes) != 0)
      return false;
    buf.base = ptr;
    buf.data = static_cast<int8_t *>(ptr);
    return true;
  }

  void *ptr = nullptr;
  if (sdkl_npu_alloc(bytes, &ptr) != 0 || ptr == nullptr)
    return false;
  buf.base = ptr;
  buf.data = static_cast<int8_t *>(ptr);
  return true;
}

void freeBuffer(Buffer &buf) {
  if (buf.base == nullptr)
    return;

  if (buf.kind == BufferKind::NPU)
    sdkl_npu_free(buf.base);
  else
    std::free(buf.base);

  buf.base = nullptr;
  buf.data = nullptr;
}

int runCase(const Options &opt, BufferKind kind, const std::vector<int8_t> &src,
            std::vector<int8_t> &converted) {
  Buffer buf;
  const size_t bytes = src.size() * sizeof(int8_t);
  if (!allocateBuffer(kind, bytes, buf)) {
    std::fprintf(stderr, "FAIL %s: allocation failed for %zu bytes\n",
                 toString(kind), bytes);
    return 2;
  }

  std::memcpy(buf.data, src.data(), bytes);
  std::fprintf(
    stderr, "CASE %s: N=%zu K=%zu bytes=%zu align_mod64=%" PRIuPTR " init=%s\n",
    toString(kind), opt.N, opt.K, bytes,
    reinterpret_cast<uintptr_t>(buf.data) % 64u, opt.init_npu ? "yes" : "no");
  std::fflush(stderr);

  const int rc =
    sdkl_cpu_rm_to_wh_i8_inplace(opt.N, opt.K, static_cast<int8_t *>(buf.data));
  if (rc != 0) {
    std::fprintf(stderr, "  FAIL rc=%d\n", rc);
    freeBuffer(buf);
    return rc;
  }

  converted.assign(buf.data, buf.data + src.size());
  const uint64_t checksum = fnv1a64(converted.data(), bytes);
  std::fprintf(stderr, "  PASS checksum=0x%016" PRIx64 "\n", checksum);
  dumpPrefix(converted.data(), bytes);
  std::fflush(stderr);

  freeBuffer(buf);
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Options opt;
  if (!parseArgs(argc, argv, opt)) {
    printUsage(argv[0]);
    return 2;
  }

  int domain = CDSP_DOMAIN_ID;
  if (opt.init_npu) {
    const int init_rc = sdkl_npu_initialize(domain, nullptr, nullptr);
    if (init_rc != 0) {
      std::fprintf(stderr, "FAIL: sdkl_npu_initialize rc=%d\n", init_rc);
      return 1;
    }

    char version[128] = {};
    sdkl_npu_get_version(domain, version);
    std::fprintf(stderr, "NPU ready: domain=%d version=%s\n", domain, version);
  } else {
    std::fprintf(stderr, "NPU init skipped by request\n");
  }

  const std::vector<int8_t> src = makeWeights(opt.N, opt.K);
  std::vector<int8_t> baseline;
  bool baseline_set = false;
  int failures = 0;

  for (BufferKind kind : opt.cases) {
    std::vector<int8_t> converted;
    const int rc = runCase(opt, kind, src, converted);
    if (rc != 0) {
      ++failures;
      continue;
    }

    if (!baseline_set) {
      baseline = converted;
      baseline_set = true;
      continue;
    }

    if (converted != baseline) {
      std::fprintf(stderr,
                   "  FAIL %s: output mismatch vs first successful "
                   "case\n",
                   toString(kind));
      ++failures;
    }
  }

  if (opt.init_npu)
    sdkl_npu_finalize(domain);

  std::fprintf(stderr,
               failures == 0 ? "=== PASS: all requested cases matched "
                               "===\n"
                             : "=== FAIL: %d case(s) failed ===\n",
               failures);
  return failures == 0 ? 0 : 1;
}
