"""
Base class for face detectors.

As per PRD Section 5.3, this defines the common interface for all face detectors
to allow plugging in alternative detectors.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..core.types import FaceBBox, Frame


class FaceDetector(ABC):
    """
    Abstract base class for face detectors.

    All face detectors must implement this interface to be used in the pipeline.
    """

    def __init__(self, confidence_threshold: float = 0.5, device: str = "cuda"):
        """
        Initialize the face detector.

        Args:
            confidence_threshold: Minimum confidence for face detection
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self._model = None

    @abstractmethod
    def detect(self, frame: Frame) -> List[FaceBBox]:
        """
        Detect faces in a frame.

        Args:
            frame: Input image/frame (H, W, C) in BGR or RGB format

        Returns:
            List of FaceBBox objects with detection confidence
        """

    @abstractmethod
    def load_model(self) -> None:
        """Load the detection model."""

    def detect_single(self, frame: Frame) -> Optional[FaceBBox]:
        """
        Detect a single face (largest by area) in a frame.

        Args:
            frame: Input image/frame

        Returns:
            Largest detected face or None if no face found
        """
        faces = self.detect(frame)
        if not faces:
            return None

        # Return the face with largest area
        return max(faces, key=lambda f: f.width * f.height)
