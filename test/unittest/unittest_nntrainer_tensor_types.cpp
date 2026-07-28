// SPDX-License-Identifier: Apache-2.0
/**
 * @file unittest_tensor_types.cpp
 * @date 11 December 2025
 * @brief Unit tests for various tensor types (CharTensor, ShortTensor,
 * UIntTensor, Int4QTensor)
 * @see https://github.com/nntrainer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <cmath>
#include <gtest/gtest.h>
#include <random>
#include <sstream>

#include <char_tensor.h>
#include <short_tensor.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace {
constexpr float TOLERANCE = 1e-5f;
} // namespace

//==============================================================================
// CharTensor Tests (int8_t based tensor)
//==============================================================================

TEST(CharTensor, constructor_default) {
  nntrainer::CharTensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.getName(), "test");
}

TEST(CharTensor, constructor_with_dim) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  EXPECT_EQ(t.batch(), 1u);
  EXPECT_EQ(t.channel(), 1u);
  EXPECT_EQ(t.height(), 2u);
  EXPECT_EQ(t.width(), 3u);
}

TEST(CharTensor, setZero) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setZero();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(CharTensor, setValue_getValue) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 20.0f);
  t.setValue(0, 0, 1, 0, 30.0f);
  t.setValue(0, 0, 1, 1, 40.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 10);
  EXPECT_EQ(t.getValue(0, 0, 0, 1), 20);
  EXPECT_EQ(t.getValue(0, 0, 1, 0), 30);
  EXPECT_EQ(t.getValue(0, 0, 1, 1), 40);
}

TEST(CharTensor, initialize_zeros) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ZEROS);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(CharTensor, initialize_ones) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ONES);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 1);
  }
}

TEST(CharTensor, argmax) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::CharTensor t(dim, true);

  // Batch 0: [10, 50, 30, 20] -> max at index 1
  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 50.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 0, 3, 20.0f);

  // Batch 1: [5, 15, 25, 100] -> max at index 3
  t.setValue(1, 0, 0, 0, 5.0f);
  t.setValue(1, 0, 0, 1, 15.0f);
  t.setValue(1, 0, 0, 2, 25.0f);
  t.setValue(1, 0, 0, 3, 100.0f);

  auto result = t.argmax();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 1u);
  EXPECT_EQ(result[1], 3u);
}

TEST(CharTensor, argmin) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::CharTensor t(dim, true);

  // Batch 0: [10, 50, 5, 20] -> min at index 2
  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 50.0f);
  t.setValue(0, 0, 0, 2, 5.0f);
  t.setValue(0, 0, 0, 3, 20.0f);

  // Batch 1: [1, 15, 25, 100] -> min at index 0
  t.setValue(1, 0, 0, 0, 1.0f);
  t.setValue(1, 0, 0, 1, 15.0f);
  t.setValue(1, 0, 0, 2, 25.0f);
  t.setValue(1, 0, 0, 3, 100.0f);

  auto result = t.argmin();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 0u);
}

TEST(CharTensor, maxValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 127.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 1, 0, -50.0f);
  t.setValue(0, 0, 1, 1, -128.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.maxValue(), 127.0f);
}

TEST(CharTensor, minValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, 127.0f);
  t.setValue(0, 0, 0, 2, 30.0f);
  t.setValue(0, 0, 1, 0, -50.0f);
  t.setValue(0, 0, 1, 1, -128.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.minValue(), -128.0f);
}

TEST(CharTensor, max_abs) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.setValue(0, 0, 0, 1, -120.0f); // abs = 120 (max)
  t.setValue(0, 0, 1, 0, 50.0f);
  t.setValue(0, 0, 1, 1, -30.0f);

  EXPECT_EQ(t.max_abs(), 120.0f);
}

TEST(CharTensor, operator_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t1(dim, true);
  nntrainer::CharTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 10.0f);
  t1.setValue(0, 0, 0, 1, 20.0f);
  t1.setValue(0, 0, 1, 0, 30.0f);
  t1.setValue(0, 0, 1, 1, 40.0f);

  t2.setValue(0, 0, 0, 0, 10.0f);
  t2.setValue(0, 0, 0, 1, 20.0f);
  t2.setValue(0, 0, 1, 0, 30.0f);
  t2.setValue(0, 0, 1, 1, 40.0f);

  EXPECT_TRUE(t1 == t2);
}

TEST(CharTensor, operator_not_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t1(dim, true);
  nntrainer::CharTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 10.0f);
  t2.setValue(0, 0, 0, 0, 99.0f);

  EXPECT_FALSE(t1 == t2);
}

TEST(CharTensor, addValue) {
  nntrainer::TensorDim dim(1, 1, 1, 1);
  nntrainer::CharTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 10.0f);
  t.addValue(0, 0, 0, 0, 5.0f, 1.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 15);
}

TEST(CharTensor, getMemoryBytes) {
  nntrainer::TensorDim dim(1, 1, 4, 4); // 16 elements
  nntrainer::CharTensor t(dim, true);

  // CharTensor uses int8_t (1 byte per element) plus scale data
  // Memory size varies based on quantization scheme, just verify it's non-zero
  EXPECT_GT(t.getMemoryBytes(), 0u);
}

TEST(CharTensor, print) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::CharTensor t(dim, true, nntrainer::Initializer::ZEROS,
                          "test_tensor");
  t.initialize();

  std::ostringstream oss;
  t.print(oss);

  std::string output = oss.str();
  // Just verify print doesn't throw and produces some output
  EXPECT_FALSE(output.empty());
}

//==============================================================================
// ShortTensor Tests (int16_t based tensor)
//==============================================================================

TEST(ShortTensor, constructor_default) {
  nntrainer::ShortTensor t("test", nntrainer::Tformat::NCHW);
  EXPECT_EQ(t.getName(), "test");
}

TEST(ShortTensor, constructor_with_dim) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  EXPECT_EQ(t.batch(), 1u);
  EXPECT_EQ(t.channel(), 1u);
  EXPECT_EQ(t.height(), 2u);
  EXPECT_EQ(t.width(), 3u);
}

TEST(ShortTensor, setZero) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setZero();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(ShortTensor, setValue_getValue) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1000.0f);
  t.setValue(0, 0, 0, 1, 2000.0f);
  t.setValue(0, 0, 1, 0, -3000.0f);
  t.setValue(0, 0, 1, 1, 32767.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 1000);
  EXPECT_EQ(t.getValue(0, 0, 0, 1), 2000);
  EXPECT_EQ(t.getValue(0, 0, 1, 0), -3000);
  EXPECT_EQ(t.getValue(0, 0, 1, 1), 32767);
}

TEST(ShortTensor, initialize_zeros) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ZEROS);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 0);
  }
}

TEST(ShortTensor, initialize_ones) {
  nntrainer::TensorDim dim(1, 1, 3, 3);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ONES);
  t.initialize();

  for (unsigned int i = 0; i < t.size(); ++i) {
    EXPECT_EQ(t.getValue(i), 1);
  }
}

TEST(ShortTensor, argmax) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::ShortTensor t(dim, true);

  // Batch 0: [100, 5000, 300, 200] -> max at index 1
  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 5000.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 0, 3, 200.0f);

  // Batch 1: [50, 150, 250, 10000] -> max at index 3
  t.setValue(1, 0, 0, 0, 50.0f);
  t.setValue(1, 0, 0, 1, 150.0f);
  t.setValue(1, 0, 0, 2, 250.0f);
  t.setValue(1, 0, 0, 3, 10000.0f);

  auto result = t.argmax();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 1u);
  EXPECT_EQ(result[1], 3u);
}

TEST(ShortTensor, argmin) {
  nntrainer::TensorDim dim(2, 1, 1, 4);
  nntrainer::ShortTensor t(dim, true);

  // Batch 0: [100, 5000, 50, 200] -> min at index 2
  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 5000.0f);
  t.setValue(0, 0, 0, 2, 50.0f);
  t.setValue(0, 0, 0, 3, 200.0f);

  // Batch 1: [10, 150, 250, 10000] -> min at index 0
  t.setValue(1, 0, 0, 0, 10.0f);
  t.setValue(1, 0, 0, 1, 150.0f);
  t.setValue(1, 0, 0, 2, 250.0f);
  t.setValue(1, 0, 0, 3, 10000.0f);

  auto result = t.argmin();
  EXPECT_EQ(result.size(), 2u);
  EXPECT_EQ(result[0], 2u);
  EXPECT_EQ(result[1], 0u);
}

TEST(ShortTensor, maxValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 32767.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 1, 0, -5000.0f);
  t.setValue(0, 0, 1, 1, -32768.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.maxValue(), 32767.0f);
}

TEST(ShortTensor, minValue) {
  nntrainer::TensorDim dim(1, 1, 2, 3);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, 32767.0f);
  t.setValue(0, 0, 0, 2, 300.0f);
  t.setValue(0, 0, 1, 0, -5000.0f);
  t.setValue(0, 0, 1, 1, -32768.0f);
  t.setValue(0, 0, 1, 2, 0.0f);

  EXPECT_EQ(t.minValue(), -32768.0f);
}

TEST(ShortTensor, max_abs) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 100.0f);
  t.setValue(0, 0, 0, 1, -12000.0f); // abs = 12000 (max)
  t.setValue(0, 0, 1, 0, 5000.0f);
  t.setValue(0, 0, 1, 1, -3000.0f);

  EXPECT_EQ(t.max_abs(), 12000.0f);
}

TEST(ShortTensor, operator_equal) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t1(dim, true);
  nntrainer::ShortTensor t2(dim, true);

  t1.setValue(0, 0, 0, 0, 1000.0f);
  t1.setValue(0, 0, 0, 1, 2000.0f);
  t1.setValue(0, 0, 1, 0, 3000.0f);
  t1.setValue(0, 0, 1, 1, 4000.0f);

  t2.setValue(0, 0, 0, 0, 1000.0f);
  t2.setValue(0, 0, 0, 1, 2000.0f);
  t2.setValue(0, 0, 1, 0, 3000.0f);
  t2.setValue(0, 0, 1, 1, 4000.0f);

  EXPECT_TRUE(t1 == t2);
}

TEST(ShortTensor, addValue) {
  nntrainer::TensorDim dim(1, 1, 1, 1);
  nntrainer::ShortTensor t(dim, true);

  t.setValue(0, 0, 0, 0, 1000.0f);
  t.addValue(0, 0, 0, 0, 500.0f, 1.0f);

  EXPECT_EQ(t.getValue(0, 0, 0, 0), 1500);
}

TEST(ShortTensor, getMemoryBytes) {
  nntrainer::TensorDim dim(1, 1, 4, 4); // 16 elements
  nntrainer::ShortTensor t(dim, true);

  // ShortTensor uses int16_t (2 bytes per element) plus scale data
  // Memory size varies based on quantization scheme, just verify it's non-zero
  EXPECT_GT(t.getMemoryBytes(), 0u);
}

TEST(ShortTensor, print) {
  nntrainer::TensorDim dim(1, 1, 2, 2);
  nntrainer::ShortTensor t(dim, true, nntrainer::Initializer::ZEROS,
                           "test_short");
  t.initialize();

  std::ostringstream oss;
  t.print(oss);

  std::string output = oss.str();
  // Just verify print doesn't throw and produces some output
  EXPECT_FALSE(output.empty());
}

//==============================================================================
// Main function
//==============================================================================

//==============================================================================
// QINT4_HTP Tests (INT4 weights for the HTP u8i4 kernel)
//
// QINT4_HTP is CharTensor-backed like QINT8, but carries PER_CHANNEL_AFFINE_I4
// so the dispatch can tell the two apart, and holds its payload WH-packed two
// values per byte. The dtype plumbing below -- element size, name, backing
// class, qscheme, qparam sizing -- had no coverage of its own; every existing
// INT4 test went through the kernel, which needs a Hexagon device.
//==============================================================================

TEST(Qint4Htp, element_size_is_one_byte) {
  // Two int4 values share a byte, but the tensor allocates one byte per
  // element and uses the first half. Sizing it at 4 bits would halve every
  // allocation and truncate the payload.
  nntrainer::TensorDim d(1, 1, 4, 32,
                         {nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT4_HTP});
  EXPECT_EQ(d.getDataTypeSize(), sizeof(int8_t));
  EXPECT_EQ(d.getDataLen(), 4u * 32u);
}

TEST(Qint4Htp, dtype_name_round_trips) {
  nntrainer::TensorDim d(1, 1, 4, 32,
                         {nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT4_HTP});
  std::stringstream ss;
  ss << d;
  EXPECT_NE(ss.str().find("QINT4_HTP"), std::string::npos);
}

TEST(Qint4Htp, is_backed_by_char_tensor) {
  nntrainer::TensorDim d(1, 1, 4, 32,
                         {nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor t(d, true);
  EXPECT_EQ(t.getDataType(), nntrainer::Tdatatype::QINT4_HTP);
  EXPECT_NE(t.getData<int8_t>(), nullptr);
}

TEST(Qint4Htp, forces_per_channel_affine_i4_qscheme) {
  // The dtype selects the scheme; a caller asking for anything else must not
  // be able to produce a tensor the u8i4 dispatch would then reject or, worse,
  // accept with the wrong qparam layout.
  nntrainer::TensorDim d(1, 1, 4, 32,
                         {nntrainer::Tformat::NCHW,
                          nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor t(d, true, nntrainer::Initializer::NONE, "",
                      nntrainer::QScheme::PER_TENSOR_AFFINE);
  EXPECT_EQ(t.q_scheme(), nntrainer::QScheme::PER_CHANNEL_AFFINE_I4);
}

TEST(Qint4Htp, qparams_are_sized_per_output_channel) {
  // One scale and one zp_corr per output channel, and the tensor is stored
  // [K, N] so that is width(). Getting this wrong gives K scales where the
  // kernel indexes N of them -- an out-of-bounds read the kernel cannot see.
  constexpr unsigned int K = 4, N = 32;
  nntrainer::TensorDim d(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor t(d, true, nntrainer::Initializer::NONE, "",
                      nntrainer::QScheme::PER_CHANNEL_AFFINE_I4);

  EXPECT_EQ(t.scale_size(), N);
  ASSERT_NE(t.getScale<float>(), nullptr);
  ASSERT_NE(t.getZpCorr<int32_t>(), nullptr);

  for (unsigned int n = 0; n < N; ++n) {
    t.getScale<float>()[n] = 0.5f + n;
    t.getZpCorr<int32_t>()[n] = -128 * (int32_t)(n + 1);
  }
  for (unsigned int n = 0; n < N; ++n) {
    EXPECT_FLOAT_EQ(t.getScale<float>()[n], 0.5f + n);
    EXPECT_EQ(t.getZpCorr<int32_t>()[n], -128 * (int32_t)(n + 1));
  }
}

TEST(Qint4Htp, packed_payload_survives_a_write_read_round_trip) {
  // The kernel reads the first N*K/2 bytes as packed nibbles. Nothing in the
  // tensor interprets them, so this only checks the buffer is byte-addressable
  // and large enough -- which is what the over-allocation is for.
  constexpr unsigned int K = 4, N = 32;
  nntrainer::TensorDim d(
    1, 1, K, N, {nntrainer::Tformat::NCHW, nntrainer::Tdatatype::QINT4_HTP});
  nntrainer::Tensor t(d, true);

  int8_t *p = t.getData<int8_t>();
  const size_t packed = (size_t)K * N / 2;
  for (size_t i = 0; i < packed; ++i)
    p[i] = static_cast<int8_t>((i * 7 + 1) & 0xFF);
  for (size_t i = 0; i < packed; ++i)
    EXPECT_EQ(p[i], static_cast<int8_t>((i * 7 + 1) & 0xFF)) << "at " << i;
}

//==============================================================================

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Failed to initialize google test" << std::endl;
  }

  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Failed to run all tests" << std::endl;
  }

  return result;
}
