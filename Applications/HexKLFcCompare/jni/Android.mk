LOCAL_PATH := $(call my-dir)

# Standalone HexKL fc_layer mm comparison (u8i8 vs u8i4 vs FP32).
# Flat ndk-build project (Android.mk + Application.mk in this folder).
#
# Two engines: the default drives the sdkl C API directly, and --engine nntr
# calls nntrainer's hmx::shgemm_u8i{4,8}_i32 -- the path a real fc_layer takes,
# including its resident weight cache. The latter needs libnntrainer.so, which
# is only declared (not included) in the source, so the sdkl path stays free of
# the nntrainer include tree.

NNTRAINER_ROOT    := $(LOCAL_PATH)/../../..
HEXKL_ADDON_ROOT  := $(HEXKL_SDK_ROOT)
ifeq ($(HEXKL_ADDON_ROOT),)
  HEXKL_ADDON_ROOT := /local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon
endif
NNTRAINER_LIB_DIR := $(NNTRAINER_ROOT)/builddir/jni/arm64-v8a

# ---- prebuilt: libnntrainer.so (provides hmx::shgemm_u8i{4,8}_i32) ----
include $(CLEAR_VARS)
LOCAL_MODULE             := nntrainer
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libnntrainer.so
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
    -I$(HEXKL_ADDON_ROOT)/include \
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DENABLE_HEXKL=1 \
    -march=armv8.2-a -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := hexkl_fc_compare.cpp
LOCAL_SHARED_LIBRARIES := sdkl_armv8 nntrainer
include $(BUILD_EXECUTABLE)
