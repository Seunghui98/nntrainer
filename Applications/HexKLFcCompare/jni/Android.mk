LOCAL_PATH := $(call my-dir)

# Standalone HexKL fc_layer mm comparison (u8i8 vs u8i4 vs FP32).
# Flat ndk-build project (Android.mk + Application.mk in this folder), mirroring
# test/unittest/jni_htp. Links the prebuilt HTP libnntrainer.so (built by
# tools/package_android.sh -Denable-htp=true) and the armv8 libsdkl.so, with
# ENABLE_HEXKL so the real sdkl_npu_mm_u8i{4,8}_i32 kernels run on the NPU.

NNTRAINER_ROOT    := $(LOCAL_PATH)/../../..
HEXKL_ADDON_ROOT  := $(HEXKL_SDK_ROOT)
ifeq ($(HEXKL_ADDON_ROOT),)
  HEXKL_ADDON_ROOT := /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon
endif
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

# ---- prebuilt: libsdkl.so (armv8, matches the u8i{4,8} kernel path) ----
include $(CLEAR_VARS)
LOCAL_MODULE             := sdkl_armv8
LOCAL_SRC_FILES          := $(HEXKL_ADDON_ROOT)/lib/armv8_android26/libsdkl.so
LOCAL_C_INCLUDES         := $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- executable: hexkl_fc_compare ----
include $(CLEAR_VARS)
LOCAL_MODULE     := hexkl_fc_compare
LOCAL_CFLAGS     := \
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -Drestrict=__restrict \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := hexkl_fc_compare.cpp
LOCAL_SHARED_LIBRARIES := nntrainer sdkl_armv8
include $(BUILD_EXECUTABLE)
