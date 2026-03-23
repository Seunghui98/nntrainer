"""
Device utilities for nntrainer benchmark.
Device interaction including device info.
"""

import subprocess


def get_device_model():
    """Get device model name from system properties."""
    try:
        cmd = ["adb", "shell", "getprop", "ro.product.model"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            device = result.stdout.strip()
            return device
    except Exception as e:
        print(f"Error reading device model: {e}")
    return "Unknown"


def get_model_size(model_path, nntr_cfg):
    """Get model file size in human-readable format from device."""
    try:
        # Get the binary file name from config
        model_file = nntr_cfg.get("model_file_name", "model.bin")
        
        # Always get from device (that's where inference runs)
        device_path = f"{model_path}/{model_file}"
        
        # Get file size via adb
        cmd = ["adb", "shell", "wc", "-c", device_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            size_bytes = int(result.stdout.strip().split()[0])
            return format_size(size_bytes)
        else:
            print(f"Warning: Could not get model size from device: {result.stderr}")
    except Exception as e:
        print(f"Error getting model size: {e}")
    return "Unknown"


def format_size(size_bytes):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KiB', 'MiB', 'GiB']:
        if size_bytes >= 1024.0:
            size_bytes /= 1024.0
        else:
            break
    return f"{size_bytes:.2f} {unit}"
