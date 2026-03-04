"""
RetinaFace detector implementation using InsightFace.

As per PRD Section 5.3, this is the default detector for high-accuracy face detection.
Supports detecting frontal and moderately profile faces (yaw up to ±45 degrees).
"""

from typing import List

import cv2

from ..core.types import FaceBBox, Frame
from .base import FaceDetector


class RetinaFaceDetector(FaceDetector):
    """
    RetinaFace detector using InsightFace's implementation.

    This detector provides:
    - High accuracy on faces with yaw up to ±45 degrees
    - Detection confidence scores
    - Bounding box coordinates
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        model_name: str = "retinaface_r50_v1",
        det_size: tuple = (640, 640),
    ):
        """
        Initialize RetinaFace detector.

        Args:
            confidence_threshold: Minimum confidence for face detection (0-1)
            device: Device to run inference on ("cuda" or "cpu")
            model_name: Name of the RetinaFace model
            det_size: Input size for detection network
        """
        super().__init__(confidence_threshold, device)
        self.model_name = model_name
        self.det_size = det_size
        self._face_analysis = None

    def load_model(self) -> None:
        """Load the RetinaFace model using InsightFace."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required for RetinaFace detection. "
                "Install with: pip install insightface"
            )

        providers = (
            ["CUDAExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self._face_analysis = FaceAnalysis(
            name=self.model_name, root="./models", providers=providers
        )
        self._face_analysis.prepare(
            ctx_id=0 if self.device == "cuda" else -1, det_size=self.det_size
        )
        self._model = self._face_analysis.det_model

    def detect(self, frame: Frame) -> List[FaceBBox]:
        """
        Detect faces in a frame.

        Args:
            frame: Input image/frame (H, W, C) in BGR format

        Returns:
            List of FaceBBox objects sorted by confidence (highest first)
        """
        if self._face_analysis is None:
            self.load_model()

        # InsightFace expects RGB
        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        # Detect faces
        faces = self._face_analysis.get(frame_rgb)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            confidence = float(face.det_score)

            if confidence < self.confidence_threshold:
                continue

            face_bbox = FaceBBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                confidence=confidence,
            )
            results.append(face_bbox)

        # Sort by confidence (highest first)
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results

    def detect_with_landmarks(self, frame: Frame) -> List[tuple]:
        """
        Detect faces and return with landmarks.

        Args:
            frame: Input image/frame

        Returns:
            List of tuples (FaceBBox, landmarks_array)
        """
        if self._face_analysis is None:
            self.load_model()

        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        faces = self._face_analysis.get(frame_rgb)

        results = []
        for face in faces:
            bbox = face.bbox.astype(int)
            confidence = float(face.det_score)

            if confidence < self.confidence_threshold:
                continue

            face_bbox = FaceBBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
                confidence=confidence,
            )

            landmarks = face.kps if hasattr(face, "kps") else None
            results.append((face_bbox, landmarks))

        return results
