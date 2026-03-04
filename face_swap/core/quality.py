"""
Quality validation and fallback logic.

As per PRD Section 6.2, the system must handle cases where faces are
partially occluded, low resolution, or briefly out of frame by:
  - Falling back to the original frame when swap quality is below a threshold.
  - Avoiding obviously broken frames (distorted faces) by quality checks.
  - Providing clear error codes and logs for integration debugging.
"""

import logging
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import Optional, Tuple

import cv2
import numpy as np

from .types import FaceBBox, SwapResult

logger = logging.getLogger("face_swap.quality")


class QualityCode(IntEnum):
    """Error / warning codes for the quality gate.

    Integrators can switch on these codes in their own logs to
    understand why a particular frame was skipped or degraded.
    """

    OK = 0
    LOW_DETECTION_CONFIDENCE = auto()
    FACE_TOO_SMALL = auto()
    FACE_OUT_OF_BOUNDS = auto()
    SWAP_BLURRY = auto()
    SWAP_COLOR_DRIFT = auto()
    EMBEDDING_MISMATCH = auto()
    FALLBACK_ORIGINAL = auto()


@dataclass
class QualityReport:
    """Detailed quality report for a single swap operation."""

    code: QualityCode = QualityCode.OK
    score: float = 1.0
    sharpness: float = 0.0
    color_diff: float = 0.0
    face_area_ratio: float = 0.0
    message: str = ""

    @property
    def passed(self) -> bool:
        return self.code == QualityCode.OK


class QualityValidator:
    """
    Validates swap quality and decides whether to use the swap or fall back.

    As per PRD Section 6.2 this avoids obviously broken frames while
    providing clear error codes and logs.
    """

    def __init__(
        self,
        min_face_size: int = 32,
        min_detection_confidence: float = 0.4,
        min_sharpness: float = 15.0,
        max_color_diff: float = 80.0,
        min_quality_score: float = 0.3,
    ):
        """
        Args:
            min_face_size: Minimum width/height in pixels for a face.
            min_detection_confidence: Below this the detection is ignored.
            min_sharpness: Laplacian variance threshold for blur detection.
            max_color_diff: Maximum average LAB ΔE between swapped and target.
            min_quality_score: Overall score below which swap is rejected.
        """
        self.min_face_size = min_face_size
        self.min_detection_confidence = min_detection_confidence
        self.min_sharpness = min_sharpness
        self.max_color_diff = max_color_diff
        self.min_quality_score = min_quality_score

    # ----- pre-swap checks (on detection / alignment inputs) -----

    def validate_detection(
        self,
        bbox: FaceBBox,
        frame_shape: Tuple[int, int],
    ) -> QualityReport:
        """Pre-swap check on a detected face."""

        h, w = frame_shape[:2]

        # Confidence gate
        if bbox.confidence < self.min_detection_confidence:
            msg = (
                f"Detection confidence {bbox.confidence:.2f} "
                f"below threshold {self.min_detection_confidence}"
            )
            logger.debug(msg)
            return QualityReport(
                code=QualityCode.LOW_DETECTION_CONFIDENCE,
                score=bbox.confidence,
                message=msg,
            )

        # Size gate
        face_w, face_h = bbox.width, bbox.height
        if face_w < self.min_face_size or face_h < self.min_face_size:
            msg = (
                f"Face size {int(face_w)}×{int(face_h)} "
                f"below minimum {self.min_face_size}"
            )
            logger.debug(msg)
            return QualityReport(
                code=QualityCode.FACE_TOO_SMALL,
                score=0.0,
                message=msg,
            )

        # Bounds gate  (face mostly outside frame)
        visible_x1 = max(bbox.x1, 0)
        visible_y1 = max(bbox.y1, 0)
        visible_x2 = min(bbox.x2, w)
        visible_y2 = min(bbox.y2, h)
        visible_area = max(0, visible_x2 - visible_x1) * max(0, visible_y2 - visible_y1)
        total_area = face_w * face_h
        ratio = visible_area / total_area if total_area > 0 else 0
        if ratio < 0.5:
            msg = f"Face {ratio:.0%} out of frame"
            logger.debug(msg)
            return QualityReport(
                code=QualityCode.FACE_OUT_OF_BOUNDS,
                score=ratio,
                face_area_ratio=ratio,
                message=msg,
            )

        return QualityReport(score=1.0, face_area_ratio=ratio)

    # ----- post-swap checks -----

    def validate_swap(
        self,
        swap_result: SwapResult,
        original_face: Optional[np.ndarray] = None,
    ) -> QualityReport:
        """Post-swap quality gate.

        Args:
            swap_result: The result from the swap model.
            original_face: Optional aligned target face for comparison.

        Returns:
            QualityReport with pass/fail decision and diagnostics.
        """
        face = swap_result.swapped_face

        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.min_sharpness:
            msg = f"Swapped face blurry (sharpness={sharpness:.1f} < {self.min_sharpness})"
            logger.debug(msg)
            return QualityReport(
                code=QualityCode.SWAP_BLURRY,
                score=sharpness / self.min_sharpness,
                sharpness=sharpness,
                message=msg,
            )

        # 2. Colour drift (only if we have the original to compare)
        color_diff = 0.0
        if original_face is not None:
            color_diff = self._compute_color_diff(face, original_face)
            if color_diff > self.max_color_diff:
                msg = (
                    f"Colour drift too large (ΔE={color_diff:.1f} "
                    f"> {self.max_color_diff})"
                )
                logger.debug(msg)
                return QualityReport(
                    code=QualityCode.SWAP_COLOR_DRIFT,
                    score=max(0, 1 - color_diff / self.max_color_diff),
                    color_diff=color_diff,
                    sharpness=sharpness,
                    message=msg,
                )

        # 3. Model-reported quality
        if swap_result.quality_score is not None:
            if swap_result.quality_score < self.min_quality_score:
                msg = (
                    f"Model quality score {swap_result.quality_score:.2f} "
                    f"below {self.min_quality_score}"
                )
                logger.debug(msg)
                return QualityReport(
                    code=QualityCode.FALLBACK_ORIGINAL,
                    score=swap_result.quality_score,
                    sharpness=sharpness,
                    color_diff=color_diff,
                    message=msg,
                )

        return QualityReport(
            score=max(swap_result.quality_score or 1.0, 0),
            sharpness=sharpness,
            color_diff=color_diff,
        )

    def should_fallback(self, report: QualityReport) -> bool:
        """Convenience: returns True if the swap should be discarded."""
        return not report.passed

    # ----- helpers -----

    @staticmethod
    def _compute_color_diff(a: np.ndarray, b: np.ndarray) -> float:
        """Mean CIE ΔE between two BGR images (resized to match)."""
        target_size = (min(a.shape[1], b.shape[1]), min(a.shape[0], b.shape[0]))
        a_resized = cv2.resize(a, target_size)
        b_resized = cv2.resize(b, target_size)

        a_lab = cv2.cvtColor(a_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        b_lab = cv2.cvtColor(b_resized, cv2.COLOR_BGR2LAB).astype(np.float32)

        diff = np.sqrt(np.sum((a_lab - b_lab) ** 2, axis=-1))
        return float(diff.mean())
