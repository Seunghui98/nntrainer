LOCAL_PATH := $(call my-dir)

# End-to-end fc_layer check: a real nntrainer FullyConnectedLayer run twice
# over identical weights, FP32 and QINT4_HTP, against a CPU reference written
# from the quantization spec.
#
# Unlike HexKLFcCompare this one does need the nntrainer include tree: it
# builds and forwards an actual model, so it uses NeuralNetwork, Tensor and the
# ml::train layer factory rather than declaring a couple of kernel entry points
# by hand.

# abspath so the project can be built from anywhere, not just this directory.
NNTRAINER_ROOT    := $(abspath $(LOCAL_PATH)/../../..)
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
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/arm/kai \
    $(NNTRAINER_ROOT)/nntrainer/tensor/cpu_backend/fallback \
    $(NNTRAINER_ROOT)/nntrainer/utils \
    $(NNTRAINER_ROOT)/nntrainer/layers \
    $(NNTRAINER_ROOT)/nntrainer/models \
    $(NNTRAINER_ROOT)/nntrainer/compiler \
    $(NNTRAINER_ROOT)/nntrainer/graph \
    $(NNTRAINER_ROOT)/nntrainer/dataset \
    $(NNTRAINER_ROOT)/nntrainer/optimizers \
    $(NNTRAINER_ROOT)/api \
    $(NNTRAINER_ROOT)/api/ccapi/include \
    $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- prebuilt: libccapi-nntrainer.so ----
# ml::train::layer::FullyConnected / createOptimizer live here, not in
# libnntrainer.so, and building a model through the factory needs them.
include $(CLEAR_VARS)
LOCAL_MODULE             := ccapi-nntrainer
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libccapi-nntrainer.so
LOCAL_C_INCLUDES         := \
    $(NNTRAINER_ROOT)/api \
    $(NNTRAINER_ROOT)/api/ccapi/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- prebuilt: libsdkl.so ----
include $(CLEAR_VARS)
LOCAL_MODULE             := sdkl
LOCAL_SRC_FILES          := $(NNTRAINER_LIB_DIR)/libsdkl.so
LOCAL_C_INCLUDES         := $(HEXKL_ADDON_ROOT)/include
LOCAL_EXPORT_C_INCLUDES  := $(LOCAL_C_INCLUDES)
include $(PREBUILT_SHARED_LIBRARY)

# ---- executable: hexkl_fc_e2e ----
include $(CLEAR_VARS)
LOCAL_MODULE     := hexkl_fc_e2e
LOCAL_CFLAGS     := \
    -I$(HEXKL_ADDON_ROOT)/../../incs \
    -I$(HEXKL_ADDON_ROOT)/../../incs/stddef \
    -pthread -fexceptions \
    -DMIN_CPP_VERSION=201703L \
    -DENABLE_FP16=1 -DUSE__FP16=1 \
    -DENABLE_HEXKL=1 \
    -march=armv8.2-a+fp16+dotprod+i8mm -O2
LOCAL_CXXFLAGS   += -std=c++17 -frtti -fexceptions
LOCAL_LDLIBS     := -llog
LOCAL_SRC_FILES  := hexkl_fc_e2e.cpp
LOCAL_SHARED_LIBRARIES := nntrainer ccapi-nntrainer sdkl
include $(BUILD_EXECUTABLE)
