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
include $(CLEAR_VARS)
LOCAL_MODULE             := sdkl_armv8
LOCAL_SRC_FILES          := $(HEXKL_ADDON_ROOT)/lib/armv8_android26/libsdkl.so
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
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 \
    -march=armv9.2-a+fp16+nosve+nosve2 -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := $(NNTRAINER_ROOT)/test/unittest/unittest_nntrainer_htp_backend.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer sdkl
LOCAL_STATIC_LIBRARIES := gtest
include $(BUILD_EXECUTABLE)

# ---- executable: unittest_nntrainer_htp_kernel_math ----
# Host-side math around the NPU GEMM: the quantize/zp_corr/dequant reference
# checks. These build on any host, but the arm64 target is where they run
# against the same compiler and float behaviour the device path uses.
include $(CLEAR_VARS)
LOCAL_MODULE     := unittest_nntrainer_htp_kernel_math
LOCAL_CFLAGS     := \
    -I$(GTEST_ROOT)/include \
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -DENABLE_TEST=1 -DREDUCE_TOLERANCE=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := $(NNTRAINER_ROOT)/test/unittest/unittest_nntrainer_htp_kernel_math.cpp
LOCAL_SHARED_LIBRARIES := nntrainer sdkl
LOCAL_STATIC_LIBRARIES := gtest
include $(BUILD_EXECUTABLE)

# ---- executable: unittest_nntrainer_htp_kernels (direct sdkl kernel tests, armv8 libsdkl) ----
include $(CLEAR_VARS)
LOCAL_MODULE     := unittest_nntrainer_htp_kernels
LOCAL_CFLAGS     := \
    -I$(GTEST_ROOT)/include \
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -DENABLE_TEST=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := $(NNTRAINER_ROOT)/test/unittest/unittest_nntrainer_htp_kernels.cpp
LOCAL_SHARED_LIBRARIES := nntrainer sdkl_armv8
LOCAL_STATIC_LIBRARIES := gtest
include $(BUILD_EXECUTABLE)
