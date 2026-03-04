"""
Core types, utilities, and infrastructure for the face swap pipeline.
"""

from .config_loader import load_config, load_face_swap_config, load_pipeline_config
from .model_manager import ModelInfo, ModelManager
from .profiler import BenchmarkReport, PipelineProfiler, StageTimings
from .quality import QualityCode, QualityReport, QualityValidator
from .types import (
    AlignedFace,
    Embedding,
    FaceBBox,
    Frame,
    Landmarks,
    PipelineResult,
    Point,
    SwapResult,
)

__all__ = [
    # Types
    "Point",
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
    "PipelineResult",
    "Frame",
    # Quality
    "QualityValidator",
    "QualityCode",
    "QualityReport",
    # Profiler
    "PipelineProfiler",
    "StageTimings",
    "BenchmarkReport",
    # Model management
    "ModelManager",
    "ModelInfo",
    # Config
    "load_config",
    "load_pipeline_config",
    "load_face_swap_config",
]
