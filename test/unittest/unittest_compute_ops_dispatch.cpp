// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   unittest_compute_ops_dispatch.cpp
 * @date   25 April 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Verify that a per-Context ComputeOps installed via ContextData
 *         actually reaches Tensor::dot / multiply / add through the
 *         tensor's attached ContextData. End-to-end check that
 *         vendor backend dispatch works with virtual dispatch (no
 *         preprocessor branches at the call site).
 */

#include <compute_ops.h>
#include <context_data.h>
#include <gtest/gtest.h>
#include <tensor.h>

#include <atomic>
#include <memory>

#if defined(ENABLE_FP16) || defined(ENABLE_HEXKL)
#include <cpu_ops_table.h>
#endif
#ifdef ENABLE_HEXKL
#include <htp_backend.h>
#endif

namespace {

/**
 * @brief Per-test counters incremented by the mock subclass.
 *
 * The mock forwards each op to the real (CPU) backend so that result
 * correctness is preserved; the counters confirm dispatch reached the
 * mock and not the global singleton directly.
 */
struct CallCounters {
  std::atomic<int> sgemm{0};
  std::atomic<int> sgemv{0};
  std::atomic<int> ele_mul{0};
  std::atomic<int> ele_add{0};
  std::atomic<int> scopy{0};
};

/**
 * @brief Mock ComputeOps subclass: forwards to a "real" backend
 *        (the global one) for correctness while bumping per-op counters.
 *
 * Because every base-class default just throws, this only overrides
 * the ops the tests exercise. Anything the test setup happens to
 * trigger that's not overridden here would throw — the tests stay
 * inside the overridden subset (sgemm/sgemv/ele_mul/ele_add/scopy).
 */
class MockComputeOps : public nntrainer::ComputeOps {
public:
  MockComputeOps(nntrainer::ComputeOps *real, CallCounters *c) :
    real_(real), counters_(c) {}

  void sgemm_fp32(unsigned int o, bool tA, bool tB, unsigned int M,
                  unsigned int N, unsigned int K, float a, const float *A,
                  unsigned int lda, const float *B, unsigned int ldb, float b,
                  float *C, unsigned int ldc) override {
    counters_->sgemm++;
    real_->sgemm_fp32(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }
  void sgemv_fp32(unsigned int o, bool tA, unsigned int M, unsigned int N,
                  float a, const float *A, unsigned int lda, const float *X,
                  unsigned int iX, float b, float *Y,
                  unsigned int iY) override {
    counters_->sgemv++;
    real_->sgemv_fp32(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }
  void ele_mul_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_mul++;
    real_->ele_mul_fp32(N, X, Y, Z, a, b, is, os);
  }
  void ele_add_fp32(unsigned int N, const float *X, const float *Y, float *Z,
                    float a, float b, unsigned int is,
                    unsigned int os) override {
    counters_->ele_add++;
    real_->ele_add_fp32(N, X, Y, Z, a, b, is, os);
  }
  void scopy_fp32(unsigned int N, const float *X, unsigned int iX, float *Y,
                  unsigned int iY) override {
    counters_->scopy++;
    real_->scopy_fp32(N, X, iX, Y, iY);
  }

private:
  nntrainer::ComputeOps *real_;
  CallCounters *counters_;
};

/**
 * @brief Test fixture wiring a MockComputeOps into a fresh ContextData for
 *        each test, so dispatch tests can assert on per-call counters.
 */
class ComputeOpsDispatchTest : public ::testing::Test {
protected:
  void SetUp() override {
    counters = std::make_unique<CallCounters>();
    nntrainer::ensureComputeOps();
    mock_ops = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                counters.get());
    ct_data = std::make_shared<nntrainer::ContextData>();
    ct_data->setComputeOps(mock_ops.get());
  }

  std::unique_ptr<CallCounters> counters;
  std::unique_ptr<MockComputeOps> mock_ops;
  std::shared_ptr<nntrainer::ContextData> ct_data;
};

} // namespace

/**
 * @brief A Tensor with no attached ContextData should fall back to the
 *        global ops; no mock counter increments.
 */
TEST_F(ComputeOpsDispatchTest, FallbackToGlobalWhenNoContextData) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  auto out = a.dot(b);

  EXPECT_EQ(counters->sgemm.load(), 0);
  EXPECT_EQ(counters->sgemv.load(), 0);
}

/**
 * @brief When a ContextData with the mock subclass is attached to a
 *        Tensor, calling .dot dispatches through the mock.
 */
TEST_F(ComputeOpsDispatchTest, DotDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);
  auto out = a.dot(b);

  EXPECT_GT(counters->sgemm.load() + counters->sgemv.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 4.0f);
}

/**
 * @brief Element-wise multiply through the attached ContextData should
 *        invoke the mock's ele_mul_fp32.
 */
TEST_F(ComputeOpsDispatchTest, MultiplyDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out);

  EXPECT_GT(counters->ele_mul.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 6.0f);
}

/**
 * @brief Element-wise add through the attached ContextData should
 *        invoke the mock's ele_add_fp32.
 */
TEST_F(ComputeOpsDispatchTest, AddDispatchesThroughAttachedContextOps) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.add(b, out);

  EXPECT_GT(counters->ele_add.load(), 0);
  EXPECT_FLOAT_EQ(out.getValue<float>(0, 0, 0, 0), 5.0f);
}

/**
 * @brief Result tensor of a binary op inherits ContextData from `this`,
 *        so a chained op on the result keeps dispatching through the
 *        same backend.
 */
TEST_F(ComputeOpsDispatchTest, ResultInheritsContextDataFromOperand) {
  nntrainer::Tensor a(1, 1, 4, 4);
  nntrainer::Tensor b(1, 1, 4, 4);
  a.setValue(1.0f);
  b.setValue(1.0f);

  a.setContextData(ct_data);

  auto first = a.dot(b);
  EXPECT_EQ(first.getContextData().get(), ct_data.get());

  int before = counters->sgemm.load() + counters->sgemv.load();
  auto second = first.dot(b);
  int after = counters->sgemm.load() + counters->sgemv.load();
  EXPECT_GT(after, before);
}

/**
 * @brief Replacing the attached ContextData with a different subclass
 *        rebinds dispatch on subsequent calls — the runtime swap
 *        property required for hot-swapping vendor contexts.
 */
TEST_F(ComputeOpsDispatchTest, SwappingContextDataRebindsDispatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out1(1, 1, 1, 8);
  nntrainer::Tensor out2(1, 1, 1, 8);

  a.setContextData(ct_data);
  a.multiply(b, out1);
  int after_first = counters->ele_mul.load();
  EXPECT_GT(after_first, 0);

  // Swap to a fresh ContextData carrying its own MockComputeOps with
  // independent counters. Subsequent ops bump fresh counters only.
  auto fresh_counters = std::make_unique<CallCounters>();
  auto fresh_mock = std::make_unique<MockComputeOps>(nntrainer::getComputeOps(),
                                                     fresh_counters.get());
  auto fresh_ct = std::make_shared<nntrainer::ContextData>();
  fresh_ct->setComputeOps(fresh_mock.get());
  a.setContextData(fresh_ct);

  a.multiply(b, out2);
  EXPECT_EQ(counters->ele_mul.load(), after_first);
  EXPECT_GT(fresh_counters->ele_mul.load(), 0);
}

/**
 * @brief Cross-vendor mismatch — when two operands of a binary op
 *        carry DIFFERENT ContextData (e.g. one CPU-resident tensor,
 *        one OpenCL-resident tensor), the op must throw rather than
 *        silently dispatch through one side's ops onto the other
 *        side's incompatible memory. This is the assertion that
 *        protects against the most insidious mixed-backend bug.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpThrowsOnContextMismatch) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  // Two distinct ContextData instances simulate two different vendor
  // contexts (e.g. CPU + OpenCL) — same kind of mock here, but the
  // identity of the ContextData pointers differs.
  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  EXPECT_THROW(a.multiply(b, out), std::invalid_argument);
  EXPECT_THROW(a.add(b, out), std::invalid_argument);
  EXPECT_THROW(a.divide(b, out), std::invalid_argument);
}

/**
 * @brief Same ContextData identity on both operands → no throw.
 *        Confirms the mismatch check is keyed on identity, not on
 *        nullness alone (which would be backward-compatibility break).
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsSameContext) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  b.setContextData(ct_data); // same instance, not a copy

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief One operand has no ContextData → permissive (legacy code
 *        path). A tensor created without ever touching ContextData
 *        falls back to the global ops table; binary-op'ing it with
 *        a context-attached tensor must NOT throw — that would break
 *        every existing test and call site.
 */
TEST_F(ComputeOpsDispatchTest, BinaryOpAcceptsOneSideUnattached) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  a.setContextData(ct_data);
  // b has no ContextData — legacy code path

  EXPECT_NO_THROW(a.multiply(b, out));
}

/**
 * @brief Tensor::to(target_ct) deep-copies and re-tags. Result owns
 *        the new ContextData; original is unchanged. After to(), a
 *        previously-mismatched binary op on the migrated tensor
 *        succeeds.
 */
TEST_F(ComputeOpsDispatchTest, ToMigratesContextDataAndUnblocksOp) {
  nntrainer::Tensor a(1, 1, 1, 8);
  nntrainer::Tensor b(1, 1, 1, 8);
  a.setValue(2.0f);
  b.setValue(3.0f);
  nntrainer::Tensor out(1, 1, 1, 8);

  auto ct_other = std::make_shared<nntrainer::ContextData>();
  ct_other->setComputeOps(mock_ops.get());

  a.setContextData(ct_data);
  b.setContextData(ct_other);

  // Migrate b onto a's context. Original b stays on ct_other.
  nntrainer::Tensor b_migrated = b.to(ct_data);
  EXPECT_EQ(b_migrated.getContextData().get(), ct_data.get());
  EXPECT_EQ(b.getContextData().get(), ct_other.get()); // unchanged

  // Now a.multiply(b_migrated) is on the same context — no throw.
  EXPECT_NO_THROW(a.multiply(b_migrated, out));
}

#ifdef ENABLE_HEXKL
/**
 * @brief Phase 0 smoke: the HTP ops singleton is constructible.
 */
TEST(HtpBackendSmoke, GetOpsNonNull) {
  EXPECT_NE(nntrainer::get_htp_ops(), nullptr);
}

/**
 * @brief Phase 0 smoke: the NPU either initialized or gracefully
 *        disabled, and supports_*() exactly tracks that state so a
 *        disabled backend falls back to CPU. The CPU backend itself
 *        never advertises shgemm acceleration.
 */
TEST(HtpBackendSmoke, SupportsTracksEnabledAndCpuHasNoAccel) {
  auto &be = nntrainer::HtpBackend::global();
  auto *htp = nntrainer::get_htp_ops();
#ifdef ENABLE_FP16
  EXPECT_EQ(htp->supports_shgemm(), be.enabled());
  if (!be.enabled())
    EXPECT_FALSE(htp->supports_shgemm());
  nntrainer::ensureComputeOps();
  EXPECT_FALSE(nntrainer::getComputeOps()->supports_shgemm());
#else
  (void)be;
  (void)htp;
#endif
}
#ifdef ENABLE_FP16
/**
 * @brief A tensor carrying mock ops with supports_shgemm()=true must reach
 *        the shgemm override on dotFloat32Float16's GEMM path.
 *        On x86 the mock forwards to the CPU shgemm, so the result is
 *        checked for accuracy as well.
 */
struct ShgemmCounters {
  std::atomic<int> shgemm{0};
};

/**
 * @brief CpuComputeOps subclass that advertises shgemm support and forwards
 *        the call to the real CPU shgemm, incrementing ShgemmCounters.
 */
class MockShgemmOps : public nntrainer::CpuComputeOps {
public:
  MockShgemmOps(ShgemmCounters *c) : counters_(c) {}

  bool supports_shgemm() const override { return true; }

  void shgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const float *A, unsigned int lda,
              const _FP16 *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    counters_->shgemm++;
    // Forward to CPU for correctness on x86
    nntrainer::shgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b, C, ldc);
  }

private:
  ShgemmCounters *counters_;
};

TEST(ShgemmGatingTest, SupportsTrueRoutesToShgemmOverride) {
  ShgemmCounters cnt;
  auto mock = std::make_unique<MockShgemmOps>(&cnt);
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(mock.get());

  // FP32 activation(4x8) · FP16 weight(4x8)^T => (4x4) GEMM shape (M=4,N=4,K=8)
  nntrainer::Tensor act(1, 1, 4, 8); // FP32
  nntrainer::TensorDim wdim({1, 1, 4, 8}, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor wgt(wdim);
  act.setRandNormal(0.0f, 1.0f);
  wgt.setRandNormal(0.0f, 1.0f);

  act.setContextData(ct);
  EXPECT_NO_THROW(
    act.dot(wgt, false, true)); // trans_in=true -> weight is [N,K]
  EXPECT_GT(cnt.shgemm.load(), 0)
    << "shgemm override must be called when supported";
}

TEST(ShgemmGatingTest, SupportsFalseSkipsShgemmOverride) {
  // Default CpuComputeOps: supports_shgemm()=false
  // dotFloat32Float16 must NOT call the shgemm override.
  ShgemmCounters cnt;

  // Use fresh ctx with plain CpuComputeOps (supports_shgemm=false)
  auto real_cpu = std::make_unique<nntrainer::CpuComputeOps>();
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(real_cpu.get());

  nntrainer::Tensor act(1, 1, 4, 8);
  nntrainer::TensorDim wdim({1, 1, 4, 8}, nntrainer::Tdatatype::FP16);
  nntrainer::Tensor wgt(wdim);
  act.setRandNormal(0.0f, 1.0f);
  wgt.setRandNormal(0.0f, 1.0f);

  act.setContextData(ct);
  EXPECT_NO_THROW(act.dot(wgt, false, true));
  // cnt.shgemm must be 0 — gating must route to nntrainer::shgemm() directly
  EXPECT_EQ(cnt.shgemm.load(), 0)
    << "shgemm override must NOT be called when not supported";
}
#endif // ENABLE_FP16
#endif // ENABLE_HEXKL

#ifdef ENABLE_FP16
/**
 * @brief Decode (M==1) routing must reach the shgemm override whenever the
 *        attached ComputeOps advertises NPU support and N is 32-aligned —
 *        mirroring prefill's unconditional routing. No env var, no opt-in.
 */
struct DecodeCounters {
  std::atomic<int> shgemm{0};
  std::atomic<int> hsgemv{0};
};

/**
 * @brief CpuComputeOps subclass used to verify decode (M==1) routes to the
 *        shgemm override rather than hsgemv, incrementing DecodeCounters.
 */
class MockDecodeOps : public nntrainer::CpuComputeOps {
public:
  MockDecodeOps(DecodeCounters *c) : counters_(c) {}

  bool supports_shgemm() const override { return true; }

  void shgemm(unsigned int o, bool tA, bool tB, unsigned int M, unsigned int N,
              unsigned int K, float a, const float *A, unsigned int lda,
              const _FP16 *B, unsigned int ldb, float b, float *C,
              unsigned int ldc) override {
    counters_->shgemm++;
    nntrainer::CpuComputeOps::shgemm(o, tA, tB, M, N, K, a, A, lda, B, ldb, b,
                                     C, ldc);
  }

  void hsgemv(unsigned int o, bool tA, unsigned int M, unsigned int N, float a,
              const _FP16 *A, unsigned int lda, const float *X, unsigned int iX,
              float b, float *Y, unsigned int iY) override {
    counters_->hsgemv++;
    nntrainer::CpuComputeOps::hsgemv(o, tA, M, N, a, A, lda, X, iX, b, Y, iY);
  }

private:
  DecodeCounters *counters_;
};

TEST(DecodeDispatchTest, DefaultRoutesToShgemmWhenAligned) {
  DecodeCounters cnt;
  auto mock = std::make_unique<MockDecodeOps>(&cnt);
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(mock.get());

  // M=1 decode, N=32 (aligned), K=8. No env var is set anywhere in this test.
  nntrainer::TensorDim::TensorType t_type_fp16 = {nntrainer::Tformat::NCHW,
                                                  nntrainer::Tdatatype::FP16};
  nntrainer::Tensor act(1, 1, 1, 8); // FP32
  nntrainer::Tensor wgt(1, 1, 32, 8, t_type_fp16);
  act.setRandNormal(0.0f, 1.0f);
  wgt.setRandNormal(0.0f, 1.0f);

  act.setContextData(ct);
  EXPECT_NO_THROW(act.dot(wgt, false, true));
  EXPECT_GT(cnt.shgemm.load(), 0)
    << "decode must route to shgemm by default when supported+aligned";
  EXPECT_EQ(cnt.hsgemv.load(), 0);
}

TEST(DecodeDispatchTest, FallsBackToCpuWhenNMisaligned) {
  DecodeCounters cnt;
  auto mock = std::make_unique<MockDecodeOps>(&cnt);
  auto ct = std::make_shared<nntrainer::ContextData>();
  ct->setComputeOps(mock.get());

  // M=1 decode, N=8 (NOT 32-aligned) — must fall back to hsgemv even though
  // supports_shgemm() is true, same as the prefill N%32 guard.
  nntrainer::TensorDim::TensorType t_type_fp16 = {nntrainer::Tformat::NCHW,
                                                  nntrainer::Tdatatype::FP16};
  nntrainer::Tensor act(1, 1, 1, 8);
  nntrainer::Tensor wgt(1, 1, 8, 8, t_type_fp16);
  act.setRandNormal(0.0f, 1.0f);
  wgt.setRandNormal(0.0f, 1.0f);

  act.setContextData(ct);
  EXPECT_NO_THROW(act.dot(wgt, false, true));
  EXPECT_EQ(cnt.shgemm.load(), 0);
  EXPECT_GT(cnt.hsgemv.load(), 0);
}
#endif // ENABLE_FP16

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
