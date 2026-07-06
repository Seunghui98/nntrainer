// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   wh_trailer.h
 * @brief  Self-describing WH-weight trailer appended to FP16 model bins.
 *
 * Layout (appended AFTER all normal weight data):
 *   repeated entries:
 *     [u32 name_len][name bytes][u32 N][u32 K][u64 byte_len][wh bytes]
 *   footer (last 16 bytes of file):
 *     [u64 trailer_bytes][u32 count][char magic[4] = "WHF1"]
 * The footer is last so a reader can seek from EOF, confirm the magic, and
 * skip the whole trailer if absent (old / non-WH bins). trailer_bytes is the
 * total size of all entry records (not including the footer itself), which
 * lets the reader locate the start of the variable-length entries from EOF
 * alone.
 */
#ifndef __WH_TRAILER_H__
#define __WH_TRAILER_H__
#ifdef ENABLE_HEXKL

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace nntrainer {

bool isWHBakeRequested();
void setWHBakeRequested(bool v);

namespace hmx {

static constexpr char WH_TRAILER_MAGIC[4] = {'W', 'H', 'F', '1'};

struct WHTrailerEntry {
  std::string name;
  unsigned int N = 0;
  unsigned int K = 0;
  std::vector<char> wh_bytes;
};

/** @brief Append a WH trailer (entries + footer) to os at the current pos. */
void writeWHTrailer(std::ostream &os,
                    const std::vector<WHTrailerEntry> &entries);

/**
 * @brief Parse a WH trailer from the end of is.
 * @return true if a valid trailer was found and parsed; false otherwise
 *         (plain bin / malformed). Never throws; leaves out empty on false.
 */
bool readWHTrailer(std::istream &is, std::vector<WHTrailerEntry> &out);

/**
 * @brief Process-global collector used by LayerImpl::save() to accumulate
 *        WH entries while the RunLayerContext is in scope. Non-null only
 *        while a WH-baked save is in progress.
 */
extern std::vector<WHTrailerEntry> *g_wh_collector;

} // namespace hmx
} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __WH_TRAILER_H__
