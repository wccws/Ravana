"""
Core data structures for the face swap pipeline.

This module defines typed data structures for communication between pipeline stages
as specified in the PRD Section 7.1.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Point:
    """2D point with x, y coordinates."""

    x: float
    y: float


@dataclass
class FaceBBox:
    """
    Face bounding box with detection confidence.

    Attributes:
        x1: Left coordinate
        y1: Top coordinate
        x2: Right coordinate
        y2: Bottom coordinate
        confidence: Detection confidence score (0-1)
        track_id: Optional tracking ID for temporal consistency
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    track_id: Optional[int] = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Point:
        return Point(x=(self.x1 + self.x2) / 2, y=(self.y1 + self.y2) / 2)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def scale(self, scale_factor: float) -> "FaceBBox":
        """Scale the bounding box by a factor from its center."""
        center = self.center
        new_width = self.width * scale_factor
        new_height = self.height * scale_factor
        return FaceBBox(
            x1=center.x - new_width / 2,
            y1=center.y - new_height / 2,
            x2=center.x + new_width / 2,
            y2=center.y + new_height / 2,
            confidence=self.confidence,
            track_id=self.track_id,
        )


@dataclass
class Landmarks:
    """
    Facial landmarks for a detected face.

    Supports both 68-point standard landmarks and dense mesh (468 points for MediaPipe).
    """

    points: np.ndarray  # Shape: (N, 2) for N landmarks
    confidence: Optional[float] = None

    def __post_init__(self):
        if isinstance(self.points, list):
            self.points = np.array(self.points, dtype=np.float32)

    @property
    def num_points(self) -> int:
        return len(self.points)

    def get_eye_centers(self) -> Tuple[Point, Point]:
        """Get left and right eye centers (for 68 or 468 point landmarks)."""
        if self.num_points >= 68:
            # Standard 68-point landmarks
            left_eye = self.points[36:42].mean(axis=0)
            right_eye = self.points[42:48].mean(axis=0)
        elif self.num_points == 468:
            # MediaPipe Face Mesh
            left_eye = self.points[468] if len(self.points) > 468 else self.points[33]
            right_eye = self.points[473] if len(self.points) > 473 else self.points[263]
        else:
            # Fallback: use approximate positions
            left_eye = self.points[self.num_points // 3]
            right_eye = self.points[self.num_points // 2]

        return Point(x=float(left_eye[0]), y=float(left_eye[1])), Point(
            x=float(right_eye[0]), y=float(right_eye[1])
        )


@dataclass
class AlignedFace:
    """
    Aligned and cropped face ready for the swap model.

    Attributes:
        image: Cropped and aligned face image (H, W, C)
        transformation_matrix: 2x3 affine matrix to map back to original frame
        original_bbox: Original bounding box in the source frame
        landmarks: Landmarks in the aligned face coordinate system
    """

    image: np.ndarray
    transformation_matrix: np.ndarray  # 2x3 affine matrix
    original_bbox: FaceBBox
    landmarks: Optional[Landmarks] = None
    crop_size: Tuple[int, int] = (256, 256)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape


@dataclass
class Embedding:
    """
    Identity embedding vector extracted from a face.

    Attributes:
        vector: Embedding vector (typically 512-dimensional for ArcFace)
        model_name: Name of the model used to extract the embedding
        normalized: Whether the vector is L2-normalized
    """

    vector: np.ndarray
    model_name: str
    normalized: bool = False

    def __post_init__(self):
        if isinstance(self.vector, list):
            self.vector = np.array(self.vector, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return len(self.vector)

    def normalize(self) -> "Embedding":
        """Return L2-normalized embedding."""
        if self.normalized:
            return self
        norm = np.linalg.norm(self.vector)
        if norm > 0:
            return Embedding(
                vector=self.vector / norm, model_name=self.model_name, normalized=True
            )
        return self

    def cosine_similarity(self, other: "Embedding") -> float:
        """Compute cosine similarity with another embedding."""
        v1 = self.normalize().vector
        v2 = other.normalize().vector
        return float(np.dot(v1, v2))


@dataclass
class SwapResult:
    """
    Result of face swap operation.

    Attributes:
        swapped_face: The generated swapped face image
        mask: Binary or soft mask for blending
        source_embedding: Source identity embedding used
        target_aligned: Target aligned face information
        quality_score: Optional quality metric for the swap
    """

    swapped_face: np.ndarray
    mask: np.ndarray
    source_embedding: Embedding
    target_aligned: AlignedFace
    quality_score: Optional[float] = None

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.swapped_face.shape


# Type alias for frame data
Frame = np.ndarray


@dataclass
class PipelineResult:
    """
    Complete result from the face swap pipeline for a single frame.

    Attributes:
        output_frame: Final processed frame with swapped faces
        swap_results: List of individual swap results
        processing_time_ms: Time taken to process this frame
    """

    output_frame: np.ndarray
    swap_results: List[SwapResult]
    processing_time_ms: float
