// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @brief  Util helpers for the safetensors format.
 * @file   safetensors_util.h
 * @date   18 May 2026
 */

#ifndef __SAFETENSORS_UTIL_H__
#define __SAFETENSORS_UTIL_H__
#ifdef __cplusplus

#include <string>
#include <tensor_dim.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nntrainer::safetensors {

struct TensorEntry {
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  size_t offset_start;
  size_t offset_end;
};

const char *dtypeToString(ml::train::TensorDim::DataType dtype);

std::string buildHeader(const std::vector<TensorEntry> &entries);

std::unordered_map<std::string, std::pair<size_t, size_t>>
parseHeader(const std::string &json);

} // namespace nntrainer::safetensors

#endif /* __cplusplus */
#endif /* __SAFETENSORS_UTIL_H__ */
