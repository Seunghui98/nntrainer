#!/bin/bash

# Build script for CausalLM Android application
# This script builds libcausallm_core.so and nntrainer_causallm executable
set -e

# Parse options
USE_BUILD_CACHE=0
while [[ $# -gt 0 ]]; do
    case $1 in
        --cache)
            USE_BUILD_CACHE=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cache]"
            echo "  --cache  Reuse existing nntrainer builddir if available"
            exit 1
            ;;
    esac
done


# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN} $1 ${NC}"
    echo -e "${CYAN}========================================${NC}"
}

log_step() {
    echo -e "\n${YELLOW}[Step $1]${NC} $2"
    echo -e "${YELLOW}----------------------------------------${NC}"
}

# Function to check and fix artifact location
check_artifact() {
    local filename=$1
    local libs_path="libs/arm64-v8a/$filename"
    local obj_path="obj/local/arm64-v8a/$filename"

    if [ -f "$libs_path" ]; then
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size)"
        return 0
    elif [ -f "$obj_path" ]; then
        echo -e "  ${YELLOW}[WARN]${NC} $filename found in obj but not in libs. Copying..."
        mkdir -p "libs/arm64-v8a"
        cp "$obj_path" "$libs_path"
        if [ -x "$obj_path" ]; then
            chmod +x "$libs_path"
        fi
        size=$(ls -lh "$libs_path" | awk '{print $5}')
        echo -e "  ${GREEN}[OK]${NC} $filename ($size) (Copied from obj)"
        return 0
    else
        echo -e "  ${RED}[ERROR]${NC} $filename not found!"
        log_info "  Checked paths:"
        log_info "    - $libs_path"
        log_info "    - $obj_path"
        return 1
    fi
}

get_meson_build_option() {
    local option_name=$1
    local meson_options="$NNTRAINER_ROOT/builddir/meson-info/intro-buildoptions.json"
    if command -v python3 >/dev/null 2>&1 && [ -f "$meson_options" ]; then
        python3 - "$meson_options" "$option_name" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    for option in json.load(f):
        if option.get("name") == sys.argv[2]:
            print(option.get("value", ""))
            break
PY
    fi
}

find_hexkl_addon_root() {
    if [ -n "${HEXKL_ADDON_ROOT:-}" ] && [ -d "$HEXKL_ADDON_ROOT" ]; then
        echo "$HEXKL_ADDON_ROOT"
        return 0
    fi
    local parsed_root
    parsed_root="$(get_meson_build_option hexkl-sdk-root)"
    if [ -n "$parsed_root" ] && [ -d "$parsed_root" ]; then
        echo "$parsed_root"
        return 0
    fi
    local default_root="/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.4.0.1/addons/hexkl_addon"
    if [ -d "$default_root" ]; then
        echo "$default_root"
        return 0
    fi
    return 1
}

# Check if NDK path is set
if [ -z "$ANDROID_NDK" ]; then
    log_error "ANDROID_NDK is not set. Please set it to your Android NDK path."
    log_info "Example: export ANDROID_NDK=/path/to/android-ndk-r21d"
    exit 1
fi

# Set NNTRAINER_ROOT
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NNTRAINER_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export NNTRAINER_ROOT

log_header "Build CausalLM Android Application"
log_info "NNTRAINER_ROOT: $NNTRAINER_ROOT"
log_info "Build cache: $([ "$USE_BUILD_CACHE" -eq 1 ] && echo 'enabled' || echo 'disabled (default)')"
log_info "ANDROID_NDK: $ANDROID_NDK"
log_info "Working directory: $(pwd)"

# Step 1: Build nntrainer for Android if not already built
log_step "1/4" "Build nntrainer for Android"

if [ "$USE_BUILD_CACHE" -eq 1 ] && [ -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_info "Build cache enabled: reusing existing nntrainer builddir (skipping)"
else
    log_info "Building nntrainer for Android..."
    cd "$NNTRAINER_ROOT"
    if [ -d "$NNTRAINER_ROOT/builddir" ]; then
        log_info "Removing existing builddir..."
        rm -rf builddir
    fi
    ./tools/package_android.sh
fi

# Check if build was successful
if [ ! -f "$NNTRAINER_ROOT/builddir/android_build_result/lib/arm64-v8a/libnntrainer.so" ]; then
    log_error "nntrainer build failed. Please check the build logs."
    exit 1
fi
log_success "nntrainer ready"

# Step 2: Build tokenizer library if not present
log_step "2/4" "Build Tokenizer Library"

cd "$SCRIPT_DIR"
if [ ! -f "lib/libtokenizers_android_c.a" ]; then
    log_warning "libtokenizers_android_c.a not found in lib directory."
    log_info "Attempting to build tokenizer library..."
    if [ -f "build_tokenizer_android.sh" ]; then
        ./build_tokenizer_android.sh
    else
        log_error "tokenizer library not found and build script is missing."
        log_info "Please build or download the tokenizer library for Android arm64-v8a"
        log_info "and place it in: $SCRIPT_DIR/lib/libtokenizers_android_c.a"
        exit 1
    fi
else
    log_info "Tokenizer library already built (skipping)"
fi
log_success "Tokenizer library ready"

# Step 3: Prepare json.hpp if not present
log_step "3/4" "Prepare json.hpp"

if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
    log_info "json.hpp not found. Downloading..."
    # prepare_encoder.sh expects target directory as first argument and version as second
    # It copies json.hpp to ../Applications/CausalLM/ if version is 0.2
    "$NNTRAINER_ROOT/jni/prepare_encoder.sh" "$NNTRAINER_ROOT/builddir" "0.2"
    
    if [ ! -f "$SCRIPT_DIR/json.hpp" ]; then
        log_error "Failed to download json.hpp"
        exit 1
    fi
else
    log_info "json.hpp already exists (skipping)"
fi
log_success "json.hpp ready"

# Step 4: Build CausalLM (libcausallm_core.so and nntrainer_causallm)
log_step "4/4" "Build CausalLM Core (library + executable)"

cd "$SCRIPT_DIR/jni"

# Clean previous builds
rm -rf libs obj

log_info "Building with ndk-build (builds causallm_core, nntrainer_causallm, nntr_quantize, nntr_safetensors_info)..."

# HTP auto-detect: if the nntrainer builddir was configured with enable-htp=true,
# nntr_quantize MUST be built with ENABLE_HEXKL so --fc_dtype FP16_WH actually bakes
# the WH trailer. Without this, the bake silently no-ops (plain FP16, no trailer) and
# on-device HTP prefill collapses. Mirror install_android.sh's detection so quantize
# always matches libnntrainer.so's HTP config.
HEXKL_NDK_ARGS=""
htp_opt="$(get_meson_build_option enable-htp)"
if [ "${htp_opt,,}" = "true" ]; then
    hexkl_root="$(find_hexkl_addon_root || true)"
    if [ -z "$hexkl_root" ]; then
        log_error "builddir was configured with enable-htp=true but no hexkl_addon root was found."
        log_error "Set HEXKL_ADDON_ROOT to the '.../addons/hexkl_addon' directory and re-run."
        exit 1
    fi
    HEXKL_NDK_ARGS="ENABLE_HEXKL=1 HEXKL_ADDON_ROOT=$hexkl_root"
    log_info "HEXKL bake: enabled (root=$hexkl_root)"
else
    log_info "HEXKL bake: disabled (builddir enable-htp != true)"
fi

# We explicitly set paths to ensure outputs are predictable
if ndk-build $HEXKL_NDK_ARGS NDK_PROJECT_PATH=. NDK_LIBS_OUT=./libs NDK_OUT=./obj APP_BUILD_SCRIPT=./Android.mk NDK_APPLICATION_MK=./Application.mk causallm_core nntrainer_causallm  nntr_quantize nntr_safetensors_info -j $(nproc); then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

# Verify outputs
log_info "Build artifacts:"

check_artifact "libcausallm_core.so" || exit 1
check_artifact "nntrainer_causallm" || exit 1
check_artifact "nntr_quantize" || exit 1
check_artifact "nntr_safetensors_info" || exit 1

# Summary
log_header "Build Summary"
log_success "Build completed successfully!"
log_info "Output files are in: $SCRIPT_DIR/jni/libs/arm64-v8a/"
log_info "Executables:"
log_info "  - nntrainer_causallm (main application), nntr_quantize, nntr_safetensors_info"
log_info "Libraries:"
log_info "  - libcausallm_core.so (CausalLM Core library)"
log_info "  - libnntrainer.so (nntrainer library)"
log_info "  - libccapi-nntrainer.so (nntrainer C/C API)"
log_info "  - libc++_shared.so (C++ runtime)"
log_info "To build API library, run:"
log_info "  ./build_api_lib.sh"
log_info "To install and run:"
log_info "  ./install_android.sh"
