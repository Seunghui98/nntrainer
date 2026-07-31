LOCAL_PATH := $(call my-dir)

# abspath: ndk-build reports my-dir relative to the invocation directory, so a
# plain $(LOCAL_PATH)/../../.. resolves outside the repo whenever this is built
# from anywhere but this directory -- and it has to be, since the folder is not
# named "jni".
NNTRAINER_ROOT    := $(abspath $(LOCAL_PATH)/../../..)
# Resolve the HexKL addon from the environment; the fallback is only a
# convenience for the machine it was developed on.
HEXKL_ADDON_ROOT  := $(HEXKL_SDK_ROOT)
ifeq ($(HEXKL_ADDON_ROOT),)
  HEXKL_ADDON_ROOT := /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon
endif
GTEST_ROOT        := $(NNTRAINER_ROOT)/subprojects/googletest/googletest
NNTRAINER_LIB_DIR := $(NNTRAINER_ROOT)/builddir/jni/arm64-v8a

# remote.h and the rest of the FastRPC headers come from the Hexagon SDK, not
# from the HexKL addon. When the addon is installed inside an SDK tree
# ($HEXAGON_SDK_ROOT/<ver>/addons/hexkl_addon) they sit two levels up, which is
# what this used to hardcode. A standalone unzip of the 1.0.0-beta2 package has
# no SDK above it, so the path has to be found rather than assumed. Order:
# an explicit HEXKL_INCS_DIR, the addon's own incs/, $HEXAGON_SDK_ROOT/incs,
# then the in-SDK-tree layout as the last resort so existing setups are
# unaffected.
HEXKL_INCS_DIR := $(HEXKL_INCS_DIR)
ifeq ($(HEXKL_INCS_DIR),)
  ifneq ($(wildcard $(HEXKL_ADDON_ROOT)/incs/remote.h),)
    HEXKL_INCS_DIR := $(HEXKL_ADDON_ROOT)/incs
  else ifneq ($(wildcard $(HEXAGON_SDK_ROOT)/incs/remote.h),)
    HEXKL_INCS_DIR := $(HEXAGON_SDK_ROOT)/incs
  else
    HEXKL_INCS_DIR := $(HEXKL_ADDON_ROOT)/../../incs
  endif
endif

# ---- prebuilt: libnntrainer.so ----
include $(CLEAR_VARS)
LOCAL_MODULE             := nntrainer
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libnntrainer.so
LOCAL_C_INCLUDES         := \
    $(NNTRAINER_ROOT)/nntrainer \
    $(NNTRAINER_ROOT)/nntrainer/tensor \
    $(NNTRAINER_ROOT)/nntrainer/tensor/htp_backend \
    $(NNTRAINER_ROOT)/nntrainer/tensor/htp_backend/hmx_ops \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm/kai \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
    $(NNTRAINER_ROOT)/nntrainer/utils \
    $(NNTRAINER_ROOT)/nntrainer/layers \
    $(NNTRAINER_ROOT)/nntrainer/models \
    $(NNTRAINER_ROOT)/nntrainer/compiler \
    $(NNTRAINER_ROOT)/nntrainer/graph \
    $(NNTRAINER_ROOT)/nntrainer/optimizers \
    $(NNTRAINER_ROOT)/api \
    $(NNTRAINER_ROOT)/api/ccapi/include \
    $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- prebuilt: libccapi-nntrainer.so (ml::train:: layer/optimizer factory) ----
# unittest_nntrainer_htp_backend builds NeuralNetwork models via the ml::train
# API (createLayer/createOptimizer), whose symbols live here, not in
# libnntrainer.so.
include $(CLEAR_VARS)
LOCAL_MODULE             := ccapi-nntrainer
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libccapi-nntrainer.so
LOCAL_C_INCLUDES         := \
    $(NNTRAINER_ROOT)/api \
    $(NNTRAINER_ROOT)/api/ccapi/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- prebuilt: libsdkl.so (armv9, used by unittest_nntrainer_htp_backend) ----
include $(CLEAR_VARS)
LOCAL_MODULE             := sdkl
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libsdkl.so
LOCAL_C_INCLUDES         := $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- prebuilt: libsdkl_armv8.so (armv8, used by sdkl_npu_probe) ----
# The addon's lib layout changed between revisions: 1.0.0-beta1 is
#   lib/<abi>/libsdkl.so
# while 1.0.0-beta2 inserts the Hexagon SDK version it was built against,
#   lib/<sdk_version>/<abi>/libsdkl.so
# so the ABI directory alone no longer locates the library. HEXKL_LIB_SUBDIR
# is whatever sits between lib/ and libsdkl.so; check with `ls $HEXKL_SDK_ROOT/lib`
# and export e.g. HEXKL_LIB_SUBDIR=6.4.0.2/armv8_android26 for beta2. The
# default keeps beta1 working unchanged.
HEXKL_LIB_SUBDIR := $(HEXKL_LIB_SUBDIR)
ifeq ($(HEXKL_LIB_SUBDIR),)
  HEXKL_LIB_SUBDIR := armv8_android26
endif

# WARNING: `sdkl` and `sdkl_armv8` are both prebuilts whose source file is
# named libsdkl.so, so ndk-build derives the same install target for the two
# and prints
#   warning: overriding recipe for target 'obj/local/arm64-v8a/libsdkl.so'
# Only one of them is ever installed, and it is `sdkl` -- the copy meson left
# in builddir/ -- regardless of which module a binary lists in
# LOCAL_SHARED_LIBRARIES, and regardless of whether obj/ was cleaned first.
# So every consumer of `sdkl_armv8` below is in fact linking the builddir
# library. That went unnoticed while both were the same HexKL revision; it
# stops being harmless the moment they differ, and shows up as a link failure
# naming symbols the older revision does not have.
#
# The two probes avoid this by linking the addon library directly through
# LOCAL_LDLIBS instead of declaring a prebuilt module, so they are unaffected
# by whichever libsdkl.so wins the collision. The remaining consumers
# (sdkl_npu_probe, unittest_nntrainer_htp_kernels) still have the old
# behaviour and should be moved the same way, or the two prebuilts collapsed
# into one, before they are used against a mixed setup.
HEXKL_SDKL_LDLIBS := -L$(HEXKL_ADDON_ROOT)/lib/$(HEXKL_LIB_SUBDIR) -lsdkl

include $(CLEAR_VARS)
LOCAL_MODULE             := sdkl_armv8
LOCAL_SRC_FILES          := $(HEXKL_ADDON_ROOT)/lib/$(HEXKL_LIB_SUBDIR)/libsdkl.so
LOCAL_C_INCLUDES         := $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- static: googletest ----
include $(CLEAR_VARS)
LOCAL_MODULE     := gtest
LOCAL_CFLAGS     := -I$(GTEST_ROOT)/include -I$(GTEST_ROOT)
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_SRC_FILES  := $(GTEST_ROOT)/src/gtest-all.cc
include $(BUILD_STATIC_LIBRARY)

# ---- executable: unittest_nntrainer_htp_backend ----
# Use armv9.2-a to match libnntrainer.so compiled with -march=armv9.2-a+fp16+nosve+nosve2.
include $(CLEAR_VARS)
LOCAL_MODULE     := unittest_nntrainer_htp_backend
LOCAL_CFLAGS     := \
    -I$(GTEST_ROOT)/include \
    -I$(HEXKL_INCS_DIR) \
    -I$(HEXKL_INCS_DIR)/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -Drestrict=__restrict \
    -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 \
    -march=armv9.2-a+fp16+nosve+nosve2 -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := $(NNTRAINER_ROOT)/test/unittest/unittest_nntrainer_htp_backend.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer sdkl
LOCAL_STATIC_LIBRARIES := gtest
include $(BUILD_EXECUTABLE)

# ---- executable: unittest_nntrainer_htp_kernels (direct sdkl kernel tests, armv8 libsdkl) ----
include $(CLEAR_VARS)
LOCAL_MODULE     := unittest_nntrainer_htp_kernels
LOCAL_CFLAGS     := \
    -I$(GTEST_ROOT)/include \
    -I$(HEXKL_INCS_DIR) \
    -I$(HEXKL_INCS_DIR)/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -Drestrict=__restrict \
    -DENABLE_TEST=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := $(NNTRAINER_ROOT)/test/unittest/unittest_nntrainer_htp_kernels.cpp
LOCAL_SHARED_LIBRARIES := nntrainer sdkl_armv8
LOCAL_STATIC_LIBRARIES := gtest
include $(BUILD_EXECUTABLE)

# ---- executable: sdkl_backend_probe (libnntrainer, armv8 libsdkl, no gtest) ----
include $(CLEAR_VARS)
LOCAL_MODULE     := sdkl_backend_probe
LOCAL_CFLAGS     := \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -Drestrict=__restrict \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := sdkl_backend_probe.cpp
LOCAL_SHARED_LIBRARIES := nntrainer sdkl
include $(BUILD_EXECUTABLE)

# ---- executable: sdkl_npu_probe (armv8 libsdkl, no gtest, no libnntrainer) ----
include $(CLEAR_VARS)
LOCAL_MODULE     := sdkl_npu_probe
LOCAL_CFLAGS     := \
    -pthread -fexceptions \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -Drestrict=__restrict \
    -I$(HEXKL_INCS_DIR) \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := sdkl_npu_probe.cpp
LOCAL_SHARED_LIBRARIES := sdkl_armv8
include $(BUILD_EXECUTABLE)

# ---- executable: hexkl_pin_probe (armv8 libsdkl, no gtest, no libnntrainer) ----
# Answers whether one ~79 MB weight can stay resident in NPU memory while
# matmuls run, which is the gate for moving lm_head to the u8i4 HMX path.
# Same shape as sdkl_npu_probe: talks to sdkl.h directly, links nothing else.
include $(CLEAR_VARS)
LOCAL_MODULE     := hexkl_pin_probe
LOCAL_CFLAGS     := \
    -pthread \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -Drestrict=__restrict \
    -I$(HEXKL_INCS_DIR) \
    -I$(HEXKL_ADDON_ROOT)/include \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
# Link the addon library directly rather than through the sdkl_armv8 prebuilt:
# that module loses the libsdkl.so target collision described above and the
# builddir copy gets linked instead. ndk-build will not stage the .so into
# libs/ this way, which is fine -- the runbook pushes it from the addon path.
# ndk-build warns "non-system libraries in linker flags: -lsdkl" and says this
# is likely to produce an incorrect build. What it is warning about is exactly
# what is being given up on purpose: it no longer tracks the dependency or
# stages the library. Expect the warning; it is not a problem here.
LOCAL_LDLIBS     := -llog $(HEXKL_SDKL_LDLIBS)
LOCAL_SRC_FILES  := hexkl_pin_probe.c
include $(BUILD_EXECUTABLE)

# ---- executable: hexkl_gemv_probe (armv8 libsdkl, no gtest, no libnntrainer) ----
# Checks whether sdkl_npu_mm_u8i4_i32 really accepts an unaligned n_row, which
# is what beta2's doc string claims and what lm_head's accumulator size rides
# on. Uses the beta2-only sdkl_cpu_i4_rm_to_i4_wh and sdkl_npu_get_hw_info, so
# it will not link against a beta1 libsdkl.
include $(CLEAR_VARS)
LOCAL_MODULE     := hexkl_gemv_probe
LOCAL_CFLAGS     := \
    -pthread \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -Drestrict=__restrict \
    -I$(HEXKL_INCS_DIR) \
    -I$(HEXKL_ADDON_ROOT)/include \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
# Same as hexkl_pin_probe: link the addon library directly, bypassing the
# libsdkl.so prebuilt collision.
LOCAL_LDLIBS     := -llog $(HEXKL_SDKL_LDLIBS)
LOCAL_SRC_FILES  := hexkl_gemv_probe.c
include $(BUILD_EXECUTABLE)
