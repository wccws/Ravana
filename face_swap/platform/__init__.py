"""
Platform-specific support: macOS Metal, mobile (Android/iOS), device detection.
"""

from .apple import (
    AppleDeviceInfo,
    CoreMLExporter,
    MPSInferenceRuntime,
    detect_apple_device,
    get_best_device,
    setup_onnxruntime_coreml,
)
from .mobile import MobileExportConfig, MobileExporter

__all__ = [
    "detect_apple_device",
    "get_best_device",
    "AppleDeviceInfo",
    "CoreMLExporter",
    "MPSInferenceRuntime",
    "setup_onnxruntime_coreml",
    "MobileExporter",
    "MobileExportConfig",
]
