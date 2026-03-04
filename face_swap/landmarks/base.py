"""
Base class for landmark detectors.

As per PRD Section 5.4, this defines the common interface for all landmark detectors.
"""

from abc import ABC, abstractmethod

from ..core.types import FaceBBox, Frame, Landmarks


class LandmarkDetector(ABC):
    """
    Abstract base class for facial landmark detectors.

    All landmark detectors must predict at least 68 standard landmarks
    or a dense mesh sufficient for robust alignment.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the landmark detector.

        Args:
            device: Device to run inference on ("cuda" or "cpu")
        """
        self.device = device
        self._model = None

    @abstractmethod
    def detect(self, frame: Frame, bbox: FaceBBox) -> Landmarks:
        """
        Detect facial landmarks for a face in a frame.

        Args:
            frame: Input image/frame (H, W, C)
            bbox: Bounding box of the face to detect landmarks for

        Returns:
            Landmarks object with facial keypoints
        """

    @abstractmethod
    def detect_multi(self, frame: Frame, bboxes: list) -> list:
        """
        Detect landmarks for multiple faces in a frame.

        Args:
            frame: Input image/frame
            bboxes: List of FaceBBox objects

        Returns:
            List of Landmarks objects (one per bbox)
        """

    @abstractmethod
    def load_model(self) -> None:
        """Load the landmark detection model."""
