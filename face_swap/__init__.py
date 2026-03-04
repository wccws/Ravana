"""
Face Swap SDK - Real-Time Face Swapping System

A production-ready SDK for face swapping on images, pre-recorded video, and live webcam streams.
"""

__version__ = "0.2.0"
__author__ = "Face Swap Team"

from .api import FaceSwapConfig, batch_swap, start_realtime_swap, swap_image, swap_video
from .audio import AudioProcessor
from .core.config_loader import load_config, load_face_swap_config, load_pipeline_config
from .core.model_manager import ModelInfo, ModelManager
from .core.model_router import ModelProfile, ModelRouter, SceneType
from .core.profiler import BenchmarkReport, PipelineProfiler, StageTimings
from .core.quality import QualityCode, QualityReport, QualityValidator
from .core.types import (
    AlignedFace,
    Embedding,
    FaceBBox,
    Landmarks,
    PipelineResult,
    SwapResult,
)

# Phase 2 modules
from .enhancement import (
    CodeFormerEnhancer,
    EnhancementConfig,
    FaceEnhancer,
    GFPGANEnhancer,
    RealESRGANEnhancer,
    create_enhancer,
)
from .filters import ARFilterEngine, FilterGallery, FilterPreset, OverlayMode
from .pipeline import FaceSwapPipeline, PipelineConfig
from .plugins import PluginInfo, PluginRegistry, get_registry, register_plugin
from .temporal import FlowGuidedBlender, OpticalFlowConfig, OpticalFlowSmoother
from .watermark import InvisibleWatermarker, WatermarkConfig

# Optional: TensorRT optimization (only if tensorrt is installed)
try:
    from .optimization import ExportConfig, TensorRTExporter, TensorRTRuntime
except ImportError:
    TensorRTExporter = None
    ExportConfig = None
    TensorRTRuntime = None

# Optional: Native C API bindings (only if compiled library exists)
try:
    from .native import NativeFaceSwap
except (ImportError, FileNotFoundError):
    NativeFaceSwap = None

# Optional: Training (only if torch is installed)
try:
    from .training import FaceSwapTrainer, TrainingConfig, TrainingState
except ImportError:
    FaceSwapTrainer = None
    TrainingConfig = None
    TrainingState = None

__all__ = [
    # High-level API
    "swap_image",
    "swap_video",
    "start_realtime_swap",
    "batch_swap",
    "FaceSwapConfig",
    # Pipeline
    "FaceSwapPipeline",
    "PipelineConfig",
    # Types
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
    "PipelineResult",
    # Quality
    "QualityValidator",
    "QualityCode",
    "QualityReport",
    # Profiling
    "PipelineProfiler",
    "StageTimings",
    "BenchmarkReport",
    # Models
    "ModelManager",
    "ModelInfo",
    "ModelRouter",
    "ModelProfile",
    "SceneType",
    # Config
    "load_config",
    "load_pipeline_config",
    "load_face_swap_config",
    # Watermark
    "InvisibleWatermarker",
    "WatermarkConfig",
    # Enhancement (Phase 2)
    "FaceEnhancer",
    "EnhancementConfig",
    "GFPGANEnhancer",
    "RealESRGANEnhancer",
    "CodeFormerEnhancer",
    "create_enhancer",
    # Plugins (Phase 2)
    "PluginRegistry",
    "PluginInfo",
    "get_registry",
    "register_plugin",
    # AR Filters (Phase 2)
    "ARFilterEngine",
    "FilterPreset",
    "FilterGallery",
    "OverlayMode",
    # Audio (Phase 2)
    "AudioProcessor",
    # Advanced Temporal (Phase 2)
    "OpticalFlowSmoother",
    "FlowGuidedBlender",
    "OpticalFlowConfig",
    # Training (Phase 2, optional)
    "FaceSwapTrainer",
    "TrainingConfig",
    "TrainingState",
    # Optimization (optional)
    "TensorRTExporter",
    "ExportConfig",
    "TensorRTRuntime",
    # Native bindings (optional)
    "NativeFaceSwap",
]
