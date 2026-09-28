# Upstream #4334 H1: bounds-check the safetensors / BIN weight loaders — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A malformed, crafted or truncated `.safetensors` / `.bin` weight file makes `NeuralNetwork::load()` throw `std::runtime_error` instead of reading outside the file (SIGSEGV / SIGBUS / leaking process memory into weights) or allocating an attacker-chosen amount of memory.

**Architecture:** Three layers of defence, each its own commit:
1. the safetensors header parser rejects numbers that overflow and `data_offsets` with `end < start`;
2. `NeuralNetwork::load()` checks `header_size` and every used `data_offsets` entry against the real file size before anything is allocated or read;
3. the mmap alternative of `ReadSource` carries its length, so `checkedRead()` refuses any read past the mapping (this is what protects `.bin` and every quantized `read()` override), the ifstream branch gets its short-read check back, and exceptions from the loader worker threads are propagated to the caller instead of calling `std::terminate`.

**Tech Stack:** C++17, meson/ninja, gtest. No new dependencies.

**Spec:** upstream issue https://github.com/nntrainer/nntrainer/issues/4334, item **H1** (and nothing else from that checklist).

## Global Constraints

- Base: upstream `nntrainer/nntrainer` `main` @ `2d1e4974` (contains the audited `a7ea056e`). This is an **upstream PR**, not htp_moe work: branch from upstream `main`, never from `htp_moe`, and do not carry `CLAUDE.md`, `docs/htp_*`, `docs/plans/*` or `.claude/` into it.
- Branch: `security/4334-h1-safetensors-bounds` on the fork `dlwlzzero/nntrainer`, PR into `nntrainer/nntrainer:main`.
- Commits: `git commit -s`, subject `[<component>] <subject>`, trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` (AGENTS.md).
- Format: `clang-format-14` on changed lines only (`git clang-format-14 2d1e4974`). Binary at `~/.local/bin/clang-format-14`.
- Do not edit `subprojects/`. Stay cross-platform: the Windows (`_WIN32`) branches get the same fix as POSIX.
- The PR says **"Addresses H1 of #4334"**, not "Fixes #4334" — the issue is a 34-item checklist and must stay open.

## Findings from reading the code (not in the issue)

- **Loader threads swallow nothing — they terminate.** Both INFERENCE loaders (`neuralnet.cpp` BIN `load_worker` ~L1022, safetensors lambda ~L1276) run `node->read()` inside `std::thread` with no `try`. Any exception in there, including the new bounds check, calls `std::terminate`. Without Task 3's propagation the fix would only turn a SIGSEGV into an abort. It would also leak the `mmap` because `munmap` is skipped.
- **The entry size from the header is never used.** `parseHeader()` returns `(start, end - start)`, but the load path only uses `.first`. The read length always comes from the tensor (`TensorBase::read` → `bytes()`, or the quantized overrides). So the only complete guard is the read-time check in `checkedRead` (Task 3). Task 2 is the cheap early rejection with a clear message.
- **`checkFile` in the ifstream branch** has been commented out since `883a11ae` ("[Tensor] ReadSource to support ifstream & mmap"), and the commit gives no reason. Reading exactly up to EOF does not set `eofbit`, so restoring it should not break well-formed files. Task 3 runs the whole unit test suite to confirm this rather than assume it.
- Only `util_func.cpp:101` inspects the `ReadSource` variant (`std::get_if<const char *>`). Changing that alternative to a struct means the compiler flags every call site that passes a bare `char *`. Four such sites are known: `neuralnet.cpp` ~L1055, ~L1084, ~L1302, ~L1327.

## Deliberately not in this PR

- **Exact size equality (`end - start == tensor bytes`).** Quantized weights are saved as opaque U8 blobs whose size is accounted per layer (`neuralnet.cpp` ~L840-842, `layer_bytes - known`). Their entry size is therefore not obviously one tensor's `bytes()`, and a wrong equality check would reject valid quantized files. The memory-safety bug is fully closed by the read-time bound in Task 3: an entry that is too small can only make a weight read other bytes *of the same file*. Mention this in the PR as a possible follow-up.
- Other #4334 items (H2 quantized header fields, M4 FSU `mmap` in `Tensor::activate`, L5 `readString`) touch the same area but are separate checkboxes.

## File map

| File | Change |
|---|---|
| `nntrainer/utils/safetensors_util.cpp` | `readNumber` overflow check; reject `end < start` (Task 1) |
| `nntrainer/models/neuralnet.cpp` | `header_size` / entry bounds (Task 2); `ReadView` call sites, file size on Windows, exception propagation from load threads (Task 3) |
| `nntrainer/utils/util_func.h`, `.cpp` | `ReadView`, bounds check in `checkedRead`, restore `checkFile` (Task 3) |
| `test/unittest/unittest_safetensors_quantize.cpp` | parser tests (Task 1), crafted-file load tests (Task 2), truncated `.bin` test (Task 3) |
| `test/unittest/unittest_util_func.cpp` | `checkedRead` bound tests (Task 3) |

---

### Task 0: Workspace

- [ ] **Step 1: Worktree on upstream main**

```bash
cd /home/j2z0-lee/nntrainer
git fetch https://github.com/nntrainer/nntrainer main
git worktree add ../nntrainer-4334 -b security/4334-h1-safetensors-bounds FETCH_HEAD
cd ../nntrainer-4334
git submodule sync && git submodule update --init --depth 1
meson setup build -Denable-transformer=true
ninja -C build
```

- [ ] **Step 2: Baseline — the two test binaries this plan touches are green before any change**

```bash
meson test -C build unittest_safetensors_quantize unittest_util_func --print-errorlogs
```
Expected: both `OK`. If either already fails, stop and record it; do not attribute it to this work later.

---

### Task 1: Header parser rejects overflowing numbers and `end < start`

**Files:**
- Modify: `nntrainer/utils/safetensors_util.cpp` (`Scanner::readNumber` ~L221; `parseHeaderEntries` `data_offsets` branch ~L300)
- Test: `test/unittest/unittest_safetensors_quantize.cpp`

**Interfaces:**
- Produces: `safetensors::parseHeaderEntries()` / `parseHeader()` throw `std::runtime_error` for these inputs. Task 2 relies on `parseHeader()` returning only `end >= start`, so `second` (the size) never wraps.

- [ ] **Step 1: Write the failing tests** (append after `build_parse_round_trip_p`)

```cpp
/**
 * @brief data_offsets with end < start must be rejected, not turned into a
 *        size that wraps to ~2^64 (#4334 H1).
 */
TEST(SafetensorsUtil, parse_rejects_reversed_offsets_n) {
  const std::string json =
    R"({"w":{"dtype":"F32","shape":[1,1,1,4],"data_offsets":[16,0]}})";
  EXPECT_THROW(st::parseHeaderEntries(json), std::runtime_error);
  EXPECT_THROW(st::parseHeader(json), std::runtime_error);
}

/**
 * @brief A number that does not fit in size_t must be rejected, not wrap to a
 *        small value (#4334 H1).
 */
TEST(SafetensorsUtil, parse_rejects_overflowing_offset_n) {
  const std::string json =
    R"({"w":{"dtype":"F32","shape":[4],"data_offsets":[0,99999999999999999999999]}})";
  EXPECT_THROW(st::parseHeaderEntries(json), std::runtime_error);
}
```

- [ ] **Step 2: Run, verify both fail**

```bash
ninja -C build && build/test/unittest/unittest_safetensors_quantize --gtest_filter='SafetensorsUtil.parse_rejects_*'
```
Expected: both FAIL with "Expected: ... throws an exception of type std::runtime_error. Actual: it throws nothing."

- [ ] **Step 3: Implement**

In `safetensors_util.cpp`, add `#include <limits>` next to the other includes, then:

```cpp
  size_t readNumber() {
    skipWs();
    size_t v = 0;
    while (pos < src.size() && src[pos] >= '0' && src[pos] <= '9') {
      const size_t d = static_cast<size_t>(src[pos++] - '0');
      if (v > (std::numeric_limits<size_t>::max() - d) / 10)
        throw std::runtime_error("safetensors: number out of range");
      v = v * 10 + d;
    }
    return v;
  }
```

In `parseHeaderEntries`, the `data_offsets` branch becomes:

```cpp
      } else if (field == "data_offsets") {
        s.expect('[');
        e.offset_start = s.readNumber();
        s.expect(',');
        e.offset_end = s.readNumber();
        s.expect(']');
        if (e.offset_end < e.offset_start)
          throw std::runtime_error("safetensors: data_offsets end < start for '" +
                                   key + "'");
      } else {
```

- [ ] **Step 4: Run, verify pass (whole binary, not just the new tests)**

```bash
ninja -C build && meson test -C build unittest_safetensors_quantize --print-errorlogs
```
Expected: `OK`.

- [ ] **Step 5: Format and commit**

```bash
git clang-format-14 2d1e4974
git add nntrainer/utils/safetensors_util.cpp test/unittest/unittest_safetensors_quantize.cpp
git commit -s -m "[Safetensors] Reject overflowing numbers and reversed data_offsets in header" \
  -m "readNumber() wrapped silently on overflow and parseHeader() turned end < start into a size near 2^64. Both now throw. Addresses H1 of #4334." \
  -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: `load()` checks `header_size` and entry offsets against the file size

**Files:**
- Modify: `nntrainer/models/neuralnet.cpp` (`MODEL_FORMAT_SAFETENSORS` case of `NeuralNetwork::load`, ~L1203-1244)
- Test: `test/unittest/unittest_safetensors_quantize.cpp`

**Interfaces:**
- Consumes: `parseHeader()` guarantees `end >= start` (Task 1).
- Produces: every weight's `file_offset` from a safetensors file satisfies `data_base + start + (end - start) <= f_size`, and this is checked in the calling thread, before any worker starts.

- [ ] **Step 1: Test helpers.** Give `createFcNN` an execution-mode parameter (default keeps every existing caller unchanged), and add a helper that rewrites a safetensors file:

```cpp
static std::unique_ptr<nntrainer::NeuralNetwork>
createFcNN(unsigned int input_width, unsigned int units,
           const std::string &weight_dtype = "",
           ml::train::ExecutionMode mode = ml::train::ExecutionMode::TRAIN) {
  // ... body unchanged, except the last line:
  nn->initialize(mode);
  return nn;
}

/**
 * @brief Write [header_size][header_json][data] to @a path.
 */
static void writeSafetensors(const std::string &path, uint64_t header_size,
                             const std::string &header_json,
                             const std::vector<char> &data) {
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char *>(&header_size), sizeof(header_size));
  f.write(header_json.data(), static_cast<std::streamsize>(header_json.size()));
  f.write(data.data(), static_cast<std::streamsize>(data.size()));
}

/**
 * @brief Save an FP32 FC model as safetensors and split it into header / data.
 */
static void saveFcSafetensors(const std::string &path, std::string &header,
                              std::vector<char> &data) {
  auto nn = createFcNN(8, 16);
  nn->save(path, ModelFormat::MODEL_FORMAT_SAFETENSORS, DataType::NONE);
  const std::vector<char> all = readFile(path);
  uint64_t header_size = 0;
  std::memcpy(&header_size, all.data(), sizeof(header_size));
  header.assign(all.data() + sizeof(header_size), header_size);
  data.assign(all.begin() + sizeof(header_size) + header_size, all.end());
}
```
Add `#include <cstring>` for `std::memcpy`.

- [ ] **Step 2: Write the tests.** The positive control proves that INFERENCE load of an untouched file works, so the `_n` tests are known to fail for the right reason.

```cpp
/**
 * @brief Control: an untouched FP32 safetensors loads in INFERENCE (mmap) mode.
 */
TEST(SafetensorsLoad, inference_load_intact_p) {
  const std::string path = "st_h1_intact.safetensors";
  std::string header;
  std::vector<char> data;
  saveFcSafetensors(path, header, data);
  auto nn = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  EXPECT_NO_THROW(nn->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS));
  remove(path.c_str());
}

/**
 * @brief header_size larger than the file must be rejected before it is used
 *        as an allocation size (#4334 H1).
 */
TEST(SafetensorsLoad, rejects_header_size_beyond_file_n) {
  const std::string path = "st_h1_hsize.safetensors";
  std::string header;
  std::vector<char> data;
  saveFcSafetensors(path, header, data);
  writeSafetensors(path, uint64_t{1} << 60, header, data);
  auto nn = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  EXPECT_THROW(nn->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS),
               std::runtime_error);
  remove(path.c_str());
}

/**
 * @brief data_offsets pointing past the end of the file must be rejected
 *        instead of being read through the mapping (#4334 H1).
 */
TEST(SafetensorsLoad, rejects_offsets_beyond_file_n) {
  const std::string path = "st_h1_offs.safetensors";
  std::string header;
  std::vector<char> data;
  saveFcSafetensors(path, header, data);
  auto entries = st::parseHeaderEntries(header);
  for (auto &e : entries) {
    e.offset_start += 1u << 20;
    e.offset_end += 1u << 20;
  }
  const std::string bad = st::buildHeader(entries);
  writeSafetensors(path, bad.size(), bad, data);
  auto nn = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  EXPECT_THROW(nn->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS),
               std::runtime_error);
  remove(path.c_str());
}

/**
 * @brief A truncated file (data section cut in half) must be rejected
 *        (#4334 H1).
 */
TEST(SafetensorsLoad, rejects_truncated_file_n) {
  const std::string path = "st_h1_trunc.safetensors";
  std::string header;
  std::vector<char> data;
  saveFcSafetensors(path, header, data);
  data.resize(data.size() / 2);
  writeSafetensors(path, header.size(), header, data);
  auto nn = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  EXPECT_THROW(nn->load(path, ModelFormat::MODEL_FORMAT_SAFETENSORS),
               std::runtime_error);
  remove(path.c_str());
}
```

- [ ] **Step 3: Run, verify the right ones fail**

```bash
ninja -C build && build/test/unittest/unittest_safetensors_quantize --gtest_filter='SafetensorsLoad.*'
```
Expected:
- `inference_load_intact_p` PASSES. If it fails, the harness is wrong (INFERENCE initialize with an optimizer set, or the offsets), so fix the helper before going on.
- `rejects_header_size_beyond_file_n` FAILS: today it throws `std::length_error` / `std::bad_alloc`, not `std::runtime_error`.
- `rejects_offsets_beyond_file_n` and `rejects_truncated_file_n` FAIL by throwing nothing, or the binary dies with SIGSEGV / SIGBUS. Either result is the bug being demonstrated.

- [ ] **Step 4: Implement.** In `neuralnet.cpp` add `#include <filesystem>` with the standard includes. In the safetensors case, replace the block from `uint64_t header_size = 0;` through the offset-assignment loop with:

```cpp
    const uint64_t f_size = std::filesystem::file_size(f_path);

    uint64_t header_size = 0;
    st_file.read(reinterpret_cast<char *>(&header_size), sizeof(header_size));
    NNTR_THROW_IF(!st_file, std::runtime_error)
      << "Failed to read safetensors header length from: " << f_path;
    NNTR_THROW_IF(header_size > f_size - sizeof(header_size), std::runtime_error)
      << "safetensors header_size " << header_size << " exceeds file size "
      << f_size << ": " << f_path;

    std::string header_json(header_size, '\0');
    // ... unchanged: read header, close, data_base, parseHeader, ISA check ...

    // Bytes available for weight data; every entry used below must fit in it.
    const size_t data_size = f_size - data_base;

    // Assign file offsets to each weight by name
    std::unordered_set<const Tensor *> visited_st;
    for (auto iter = model_graph.cbegin(); iter != model_graph.cend(); iter++) {
      auto weights = (*iter)->getRunContext().getWeights();
      for (auto weight : weights) {
        if (!visited_st.insert(&weight->getVariableRef()).second)
          continue;
        const std::string &name = weight->getName();
        auto it = name_offset_map.find(name);
        if (it == name_offset_map.end())
          continue;
        const auto [off, len] = it->second;
        NNTR_THROW_IF(off > data_size || len > data_size - off,
                      std::runtime_error)
          << "safetensors entry '" << name << "' [" << off << ", " << off + len
          << ") exceeds data section of " << data_size << " bytes: " << f_path;
        weight->getVariableRef().setFileOffset(data_base + off);
      }
    }
```
`f_size - sizeof(header_size)` cannot underflow: the 8-byte read above already succeeded, so `f_size >= 8`. `off + len` in the message cannot overflow because `len = end - start` with `end >= start` (Task 1).

- [ ] **Step 5: Run, verify pass**

```bash
ninja -C build && meson test -C build unittest_safetensors_quantize --print-errorlogs
```
Expected: `OK` (all four `SafetensorsLoad.*` plus every pre-existing test).

- [ ] **Step 6: Format and commit**

```bash
git clang-format-14 2d1e4974
git add nntrainer/models/neuralnet.cpp test/unittest/unittest_safetensors_quantize.cpp
git commit -s -m "[Safetensors] Validate header_size and data_offsets against the file size on load" \
  -m "header_size was used as an allocation size and data_offsets as mmap offsets without comparing either to the file. A crafted or truncated file read past the mapping (SIGSEGV/SIGBUS or process memory copied into weights). Both are now checked before any allocation or read. Addresses H1 of #4334." \
  -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: `checkedRead` refuses reads past the source; load threads propagate errors

**Files:**
- Modify: `nntrainer/utils/util_func.h:43` (`ReadSource`), `:170-178` (doc), `nntrainer/utils/util_func.cpp:91-109`
- Modify: `nntrainer/models/neuralnet.cpp` — BIN `load_worker` (~L1022-1095) and the safetensors INFERENCE lambda (~L1276-1333)
- Test: `test/unittest/unittest_util_func.cpp`, `test/unittest/unittest_safetensors_quantize.cpp`

**Interfaces:**
- Consumes: nothing from Tasks 1-2. This task is independent and protects `.bin` as well.
- Produces:
  ```cpp
  namespace nntrainer {
  /** @brief A read-only view of a mapped weight file: base pointer + length. */
  struct ReadView {
    const char *data;
    size_t size;
  };
  using ReadSource = std::variant<std::ifstream *, ReadView>;
  }
  ```
  `checkedRead(ReadSource, ...)` throws `std::runtime_error(error_msg)` when a `ReadView` read would go past `size`, or when an ifstream read comes up short.

- [ ] **Step 1: Write the failing unit tests** (`unittest_util_func.cpp`; add `#include <cstdio>` and `#include <fstream>` if missing)

```cpp
/**
 * @brief checkedRead on a mapped view must stay inside the view (#4334 H1).
 */
TEST(nntrainer_util_func, checkedRead_view_bounds_n) {
  char src[16] = {0};
  char dst[16];
  nntrainer::ReadView view{src, sizeof(src)};
  EXPECT_NO_THROW(nntrainer::checkedRead(view, dst, 8, "read", 8, true));
  EXPECT_THROW(nntrainer::checkedRead(view, dst, 8, "read", 9, true),
               std::runtime_error);
  EXPECT_THROW(nntrainer::checkedRead(view, dst, 8, "read", SIZE_MAX, true),
               std::runtime_error);
  EXPECT_THROW(nntrainer::checkedRead(view, dst, 17, "read", 0, false),
               std::runtime_error);
}

/**
 * @brief checkedRead on an ifstream must report a short read (#4334 H1).
 */
TEST(nntrainer_util_func, checkedRead_stream_short_read_n) {
  const char *path = "checked_read_short.bin";
  {
    std::ofstream f(path, std::ios::binary);
    f.write("abcd", 4);
  }
  char dst[8];
  std::ifstream in(path, std::ios::binary);
  EXPECT_THROW(nntrainer::checkedRead(&in, dst, 8, "read", 0, true),
               std::runtime_error);
  std::remove(path);
}
```

And the end-to-end BIN test (`unittest_safetensors_quantize.cpp`, next to the Task 2 tests; add `#include <filesystem>`):

```cpp
/**
 * @brief A truncated .bin must make INFERENCE (mmap, multi-threaded) load
 *        throw to the caller, not read past the mapping or terminate the
 *        process from a worker thread (#4334 H1).
 */
TEST(SafetensorsLoad, bin_truncated_inference_n) {
  const std::string path = "h1_trunc.bin";
  {
    auto nn = createFcNN(8, 16);
    nn->save(path, ModelFormat::MODEL_FORMAT_BIN, DataType::NONE);
  }
  auto nn_ok = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  ASSERT_NO_THROW(nn_ok->load(path, ModelFormat::MODEL_FORMAT_BIN));

  std::filesystem::resize_file(path, std::filesystem::file_size(path) / 2);
  auto nn = createFcNN(8, 16, "", ml::train::ExecutionMode::INFERENCE);
  EXPECT_THROW(nn->load(path, ModelFormat::MODEL_FORMAT_BIN),
               std::runtime_error);
  remove(path.c_str());
}
```

- [ ] **Step 2: Run, verify they fail**

```bash
ninja -C build
```
Expected: **compile error** in `unittest_util_func.cpp` (`ReadView` does not exist). That counts as the red state for the util tests. To see the BIN test fail at runtime first, build with only that test added:

```bash
build/test/unittest/unittest_safetensors_quantize --gtest_filter='SafetensorsLoad.bin_truncated_inference_n'
```
Expected: FAIL with "throws nothing" (the reads past EOF inside the last page return zeros), or death by SIGBUS.

- [ ] **Step 3: Implement `ReadView` and the checks**

`util_func.h`:

```cpp
namespace nntrainer {
/**
 * @brief A read-only view of a mapped weight file: base pointer + length, so
 *        reads through it can be bounds-checked.
 */
struct ReadView {
  const char *data;
  size_t size;
};
using ReadSource = std::variant<std::ifstream *, ReadView>;
```
Update the `checkedRead(ReadSource ...)` doc comment: `@throw std::runtime_error if the read would go past the end of a ReadView, or if the stream read fails or comes up short.`

`util_func.cpp`:

```cpp
void checkedRead(ReadSource src, char *array, std::streamsize size,
                 const char *error_msg, size_t start_offset,
                 bool read_from_offset) {
  const size_t n = static_cast<size_t>(size);
  if (auto f = std::get_if<std::ifstream *>(&src)) {
    if (read_from_offset) {
      (*f)->seekg(start_offset, std::ios::beg);
    }
    (*f)->read(static_cast<char *>(array), size);
    checkFile(**f, error_msg);
  } else if (auto v = std::get_if<ReadView>(&src)) {
    const size_t off = read_from_offset ? start_offset : 0;
    if (off > v->size || n > v->size - off)
      throw std::runtime_error(error_msg);
    /// @todo use mmap instead memcpy to reduce peak memory
    std::memcpy(array, v->data + off, n);
  }
}
```
Note that `checkFile` takes the stream by reference, so the argument is `**f`, not the commented-out `(*f)`.

- [ ] **Step 4: Update the four call sites and propagate worker exceptions.** In both INFERENCE loaders in `neuralnet.cpp`, apply the same pattern. Add `#include <exception>` and `#include <mutex>` if not already pulled in.

Before the threads are created:

```cpp
      // A throw inside std::thread calls std::terminate; keep the first one
      // and rethrow it on the calling thread after join.
      std::exception_ptr load_error;
      std::mutex load_error_mutex;
      auto record_load_error = [&]() {
        std::lock_guard<std::mutex> lock(load_error_mutex);
        if (!load_error)
          load_error = std::current_exception();
      };
```

Worker body: wrap the per-node work in `try { ... } catch (...) { record_load_error(); }`. For the BIN `load_worker` this means wrapping the body of the `for` loop, so the other nodes still load and the threads join normally. For the safetensors per-node lambda it means wrapping the whole lambda body. In the POSIX mmap branch, keep `munmap` on the error path:

```cpp
            char *view = static_cast<char *>(mmap_ptr);
            try {
              node->read(ReadView{view, f_size}, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);
            } catch (...) {
              ::munmap(mmap_ptr, f_size);
              throw;
            }
```

In the Windows branch, get the length from the handle (the view maps the whole file) and do the same with `UnmapViewOfFile` / `CloseHandle`:

```cpp
            LARGE_INTEGER li;
            NNTR_THROW_IF(!GetFileSizeEx(hFile, &li), std::runtime_error)
              << "GetFileSizeEx failed: " << f_path;
            const size_t f_size = static_cast<size_t>(li.QuadPart);
            ...
            try {
              node->read(ReadView{view, f_size}, false, exec_mode, fsu_mode,
                         std::numeric_limits<size_t>::max(), true,
                         model_file_fd);
            } catch (...) {
              UnmapViewOfFile(view);
              CloseHandle(hMap);
              CloseHandle(hFile);
              throw;
            }
```

After the `join` loop of each loader:

```cpp
      if (load_error)
        std::rethrow_exception(load_error);
```
Then build. The compiler must not report any remaining `char *` → `ReadSource` conversion. If it reports one (for example in `Applications/`), fix it the same way with `ReadView{ptr, len}`.

- [ ] **Step 5: Run the targeted tests, then everything.** The ifstream `checkFile` has been disabled since it was introduced, so the full suite is what shows whether restoring it breaks a well-formed read path.

```bash
ninja -C build
meson test -C build unittest_util_func unittest_safetensors_quantize --print-errorlogs
meson test -C build --print-errorlogs
```
Expected: the targeted tests are `OK`, and the full suite shows no new failures compared with the Task 0 baseline. If a test starts failing inside the restored `checkFile`, **do not comment it out again**. Find which caller reads past its data, record it in the PR, and ask before changing the approach.

- [ ] **Step 6: Format and commit**

```bash
git clang-format-14 2d1e4974
git add nntrainer/utils/util_func.h nntrainer/utils/util_func.cpp nntrainer/models/neuralnet.cpp \
        test/unittest/unittest_util_func.cpp test/unittest/unittest_safetensors_quantize.cpp
git commit -s -m "[Utils] Bound mmap weight reads by the mapping length and restore the stream short-read check" \
  -m "ReadSource's mapped alternative was a bare pointer, so checkedRead() could not tell that a read ran past the mapping (truncated .bin / .safetensors, or any offset/size mismatch). It is now a {data, size} view and out-of-range reads throw. The ifstream branch's checkFile() had been commented out, which left weights with stale memory on a short read; it is restored. The INFERENCE loader threads now carry exceptions back to load() instead of calling std::terminate, and unmap on the error path. Addresses H1 of #4334." \
  -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: ASan confirmation and PR

- [ ] **Step 1: The crafted-file tests are clean under AddressSanitizer** (#4334 asks for exactly this)

```bash
meson setup build-asan -Denable-transformer=true -Db_sanitize=address -Db_lundef=false
ninja -C build-asan
meson test -C build-asan unittest_safetensors_quantize unittest_util_func --print-errorlogs
```
Expected: `OK` with no ASan report. If ASan does not work on this host, say so in the PR instead of claiming it.

- [ ] **Step 2: Push to the fork and open the PR against upstream** (use the `gh` CLI; GitHub MCP writes return 403 for this account)

```bash
git push -u origin security/4334-h1-safetensors-bounds
gh pr create -R nntrainer/nntrainer --base main \
  --head dlwlzzero:security/4334-h1-safetensors-bounds \
  --title "[Safetensors] Bounds-check weight file loading (#4334 H1)" \
  --body-file <body.md>
```
The body follows `.github/PULL_REQUEST_TEMPLATE.md` and states:
- **Addresses H1 of #4334**, not "Fixes".
- the three commits and what each closes;
- the worker-thread `std::terminate` finding, which is not in the issue;
- that exact entry-size equality is deliberately left out, and why (quantized blobs);
- the test and ASan results exactly as observed.

It ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

- [ ] **Step 3:** After the PR merges, tick the H1 box in #4334 with a link to the PR.
