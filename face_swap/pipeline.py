"""
Main face swap pipeline orchestrator.

As per PRD Section 7.1, this module coordinates all pipeline stages:
1. Face detection
2. Landmark detection
3. Face alignment
4. Identity embedding
5. Face swap generation
6. Blending
7. Temporal smoothing (for video)

Additionally integrates:
- Quality validation (PRD §6.2)
- Performance profiling (PRD §2.2)
- Invisible watermarking (PRD §6.3)
"""

import logging
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import cv2
import numpy as np

from .alignment import FaceAligner
from .blending import FaceBlender
from .core.profiler import PipelineProfiler
from .core.quality import QualityCode, QualityValidator
from .core.types import (
    AlignedFace,
    Embedding,
    FaceBBox,
    Frame,
    Landmarks,
    PipelineResult,
    SwapResult,
)
from .detection import AsyncFaceDetector, FaceDetector, RetinaFaceDetector
from .embedding import ArcFaceEmbedder, IdentityEmbedder
from .landmarks import LandmarkDetector, MediaPipeLandmarkDetector
from .swap import FaceSwapper, InSwapperModel
from .temporal import TemporalSmoother
from .watermark import InvisibleWatermarker, WatermarkConfig

logger = logging.getLogger("face_swap.pipeline")


@dataclass
class PipelineConfig:
    """Configuration for the face swap pipeline."""

    # Device
    device: str = "cuda"

    # Detection
    detection_model: str = "retinaface"
    det_confidence_threshold: float = 0.5

    # Alignment
    crop_size: int = 256

    # Swap model
    swap_model: str = "inswapper"
    swap_model_path: Optional[str] = None

    # Blending
    blend_mode: str = "alpha"  # "alpha", "poisson", "feather"
    color_correction: bool = True

    # Temporal (for video)
    enable_temporal: bool = True
    temporal_smooth_factor: float = 0.7

    # Performance
    batch_size: int = 1

    # Real-time (PRD §5.9): decouple detection from swap
    async_detection: bool = False

    # Quality gate (PRD §6.2)
    enable_quality_gate: bool = True
    min_quality_score: float = 0.3

    # Watermarking (PRD §6.3)
    watermark_config: Optional[WatermarkConfig] = None

    # Profiling (PRD §2.2)
    enable_profiling: bool = False


class FaceSwapPipeline:
    """
    Main face swap pipeline.

    Coordinates all stages of the face swap process.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()

        # Initialize components
        self.detector: Optional[FaceDetector] = None
        self.landmark_detector: Optional[LandmarkDetector] = None
        self.embedder: Optional[IdentityEmbedder] = None
        self.aligner: Optional[FaceAligner] = None
        self.swapper: Optional[FaceSwapper] = None
        self.blender: Optional[FaceBlender] = None
        self.temporal_smoother: Optional[TemporalSmoother] = None

        # New components
        self.quality_validator: Optional[QualityValidator] = None
        self.profiler: Optional[PipelineProfiler] = None
        self.watermarker: Optional[InvisibleWatermarker] = None
        self._async_detector: Optional[AsyncFaceDetector] = None

        self._initialized = False

    def initialize(self) -> None:
        """Initialize all pipeline components."""
        if self._initialized:
            return

        cfg = self.config

        # Detection
        raw_detector = RetinaFaceDetector(
            confidence_threshold=cfg.det_confidence_threshold, device=cfg.device
        )

        if cfg.async_detection:
            self._async_detector = AsyncFaceDetector(raw_detector)
            self._async_detector.start()
            self.detector = raw_detector  # keep ref for non-async paths
        else:
            self.detector = raw_detector

        # Landmarks
        self.landmark_detector = MediaPipeLandmarkDetector(
            device="cpu"  # MediaPipe runs on CPU
        )

        # Alignment
        self.aligner = FaceAligner(crop_size=(cfg.crop_size, cfg.crop_size))

        # Embedding
        self.embedder = ArcFaceEmbedder(device=cfg.device)

        # Swap model
        if cfg.swap_model == "inswapper":
            self.swapper = InSwapperModel(
                device=cfg.device,
                model_path=cfg.swap_model_path or "./models/inswapper_128.onnx",
            )
        else:
            from .swap import SimSwapModel

            self.swapper = SimSwapModel(
                device=cfg.device,
                resolution=cfg.crop_size,
                model_path=cfg.swap_model_path,
            )

        # Blender
        self.blender = FaceBlender(
            blend_mode=cfg.blend_mode, color_correction=cfg.color_correction
        )

        # Temporal smoother (for video)
        if cfg.enable_temporal:
            self.temporal_smoother = TemporalSmoother(
                smooth_factor=cfg.temporal_smooth_factor
            )

        # Quality validator (PRD §6.2)
        if cfg.enable_quality_gate:
            self.quality_validator = QualityValidator(
                min_quality_score=cfg.min_quality_score,
            )

        # Profiler (PRD §2.2)
        self.profiler = PipelineProfiler()
        self.profiler.enabled = cfg.enable_profiling

        # Watermarker (PRD §6.3)
        wm_cfg = cfg.watermark_config or WatermarkConfig()
        self.watermarker = InvisibleWatermarker(wm_cfg)

        self._initialized = True

    def cleanup(self) -> None:
        """Release resources (e.g., stop async detector thread)."""
        if self._async_detector is not None:
            self._async_detector.stop()
            self._async_detector = None

    def process_frame(
        self,
        frame: Frame,
        source_embedding: Embedding,
        return_intermediate: bool = False,
    ) -> Union[Frame, PipelineResult]:
        """
        Process a single frame through the pipeline.

        Args:
            frame: Input frame
            source_embedding: Source identity embedding
            return_intermediate: Whether to return full PipelineResult

        Returns:
            Output frame with swapped faces, or PipelineResult if return_intermediate
        """
        if not self._initialized:
            self.initialize()

        self.profiler.begin_frame()

        # 1. Face detection
        with self.profiler.stage("detection"):
            if self._async_detector is not None and self._async_detector.is_running:
                bboxes = self._async_detector.detect(frame)
            else:
                bboxes = self.detector.detect(frame)
        self.profiler.set_num_faces(len(bboxes))

        if not bboxes:
            processing_time = self.profiler.end_frame().total_ms
            if return_intermediate:
                return PipelineResult(
                    output_frame=frame.copy(),
                    swap_results=[],
                    processing_time_ms=processing_time,
                )
            return frame.copy()

        # Pre-swap quality gate
        if self.quality_validator is not None:
            valid_bboxes = []
            for bbox in bboxes:
                report = self.quality_validator.validate_detection(
                    bbox, frame.shape[:2]
                )
                if report.passed:
                    valid_bboxes.append(bbox)
                else:
                    logger.debug("Skipped face: %s", report.message)
            bboxes = valid_bboxes

        if not bboxes:
            processing_time = self.profiler.end_frame().total_ms
            if return_intermediate:
                return PipelineResult(
                    output_frame=frame.copy(),
                    swap_results=[],
                    processing_time_ms=processing_time,
                )
            return frame.copy()

        # 2. Landmark detection
        with self.profiler.stage("landmarks"):
            landmarks_list = []
            for bbox in bboxes:
                lm = self.landmark_detector.detect(frame, bbox)
                landmarks_list.append(lm)

        # 3. Face alignment
        with self.profiler.stage("alignment"):
            aligned_faces = []
            for bbox, landmarks in zip(bboxes, landmarks_list):
                if landmarks.num_points > 0:
                    aligned = self.aligner.align(frame, landmarks, bbox)
                else:
                    aligned = self.aligner.align_simple(frame, bbox)
                aligned_faces.append(aligned)

        # 4. Face swapping
        with self.profiler.stage("swap"):
            swap_results = []
            for aligned in aligned_faces:
                result = self.swapper.swap(aligned, source_embedding)
                swap_results.append(result)

        # Post-swap quality gate
        if self.quality_validator is not None:
            filtered = []
            for result in swap_results:
                report = self.quality_validator.validate_swap(result)
                if not self.quality_validator.should_fallback(report):
                    filtered.append(result)
                else:
                    logger.debug("Swap rejected: %s", report.message)
            swap_results = filtered

        # 5. Blending
        with self.profiler.stage("blend"):
            output_frame = frame.copy()
            for result in swap_results:
                output_frame = self.blender.blend(output_frame, result)

        # 6. Watermark
        with self.profiler.stage("watermark"):
            if self.watermarker.config.enabled:
                provenance = self.watermarker.create_provenance(
                    model_name=self.config.swap_model,
                )
                output_frame = self.watermarker.embed(output_frame, provenance)

        timings = self.profiler.end_frame()

        if return_intermediate:
            return PipelineResult(
                output_frame=output_frame,
                swap_results=swap_results,
                processing_time_ms=timings.total_ms,
            )

        return output_frame

    def process_video_frame(
        self,
        frame: Frame,
        source_embedding: Embedding,
        frame_number: int = 0,
    ) -> Frame:
        """
        Process a video frame with temporal smoothing.

        Args:
            frame: Input frame
            source_embedding: Source identity embedding
            frame_number: Current frame number

        Returns:
            Output frame with swapped faces
        """
        if not self._initialized:
            self.initialize()

        self.profiler.begin_frame()

        # Face detection
        with self.profiler.stage("detection"):
            if self._async_detector is not None and self._async_detector.is_running:
                bboxes = self._async_detector.detect(frame)
            else:
                bboxes = self.detector.detect(frame)
        self.profiler.set_num_faces(len(bboxes))

        if not bboxes:
            self.profiler.end_frame()
            return frame.copy()

        # Pre-swap quality gate
        if self.quality_validator is not None:
            bboxes = [
                b
                for b in bboxes
                if self.quality_validator.validate_detection(b, frame.shape[:2]).passed
            ]
        if not bboxes:
            self.profiler.end_frame()
            return frame.copy()

        # Temporal smoothing of positions
        with self.profiler.stage("temporal"):
            if self.temporal_smoother is not None:
                bboxes = self.temporal_smoother.smooth_bboxes(bboxes, frame)

        # Landmark detection
        with self.profiler.stage("landmarks"):
            landmarks_list = []
            for bbox in bboxes:
                lm = self.landmark_detector.detect(frame, bbox)
                landmarks_list.append(lm)

        # Face alignment
        with self.profiler.stage("alignment"):
            aligned_faces = []
            for bbox, landmarks in zip(bboxes, landmarks_list):
                if landmarks.num_points > 0:
                    aligned = self.aligner.align(frame, landmarks, bbox)
                else:
                    aligned = self.aligner.align_simple(frame, bbox)
                aligned_faces.append(aligned)

        # Face swapping
        with self.profiler.stage("swap"):
            swap_results = []
            for i, aligned in enumerate(aligned_faces):
                result = self.swapper.swap(aligned, source_embedding)

                # Temporal smoothing of appearance
                track_id = getattr(bboxes[i], "track_id", i)
                if self.temporal_smoother is not None:
                    result = self.temporal_smoother.smooth_swap_result(track_id, result)

                # Post-swap quality gate
                if self.quality_validator is not None:
                    report = self.quality_validator.validate_swap(result)
                    if self.quality_validator.should_fallback(report):
                        logger.debug("Video frame swap rejected: %s", report.message)
                        continue

                swap_results.append(result)

        # Blending
        with self.profiler.stage("blend"):
            output_frame = frame.copy()
            for result in swap_results:
                output_frame = self.blender.blend(output_frame, result)

        # Watermark
        with self.profiler.stage("watermark"):
            if self.watermarker.config.enabled:
                output_frame = self.watermarker.embed(output_frame)

        self.profiler.end_frame()
        return output_frame

    def extract_source_embedding(self, source_image: Frame) -> Embedding:
        """
        Extract identity embedding from a source image.

        Args:
            source_image: Source face image

        Returns:
            Identity embedding
        """
        if not self._initialized:
            self.initialize()

        # Detect face
        bbox = self.detector.detect_single(source_image)

        if bbox is None:
            raise ValueError("No face detected in source image")

        # Detect landmarks
        landmarks = self.landmark_detector.detect(source_image, bbox)

        # Align face
        aligned = self.aligner.align(source_image, landmarks, bbox)

        # Extract embedding
        with self.profiler.stage("embedding"):
            embedding = self.embedder.extract(aligned)

        return embedding

    def extract_source_embedding_multi(self, source_images: List[Frame]) -> Embedding:
        """
        Extract averaged identity embedding from multiple source images.

        Args:
            source_images: List of source face images

        Returns:
            Averaged identity embedding
        """
        if not self._initialized:
            self.initialize()

        aligned_faces = []
        for img in source_images:
            bbox = self.detector.detect_single(img)
            if bbox is None:
                continue

            landmarks = self.landmark_detector.detect(img, bbox)
            aligned = self.aligner.align(img, landmarks, bbox)
            aligned_faces.append(aligned)

        if not aligned_faces:
            raise ValueError("No faces detected in source images")

        return self.embedder.extract_average(aligned_faces)

    def get_benchmark_report(self):
        """Return a benchmark report from the profiler (PRD §2.2)."""
        return self.profiler.report()
