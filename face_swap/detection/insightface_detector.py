"""
Alternative detector using InsightFace's face analysis directly.

This provides both detection and recognition in one pass.
"""

from typing import List, Tuple

import cv2

from ..core.types import Embedding, FaceBBox, Frame, Landmarks
from .base import FaceDetector


class InsightFaceDetector(FaceDetector):
    """
    Combined detector and recognizer using InsightFace.

    This is optimized for real-time use as it can return both
    detections and embeddings in a single forward pass.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
    ):
        super().__init__(confidence_threshold, device)
        self.det_size = det_size
        self._face_analysis = None

    def load_model(self) -> None:
        """Load the InsightFace analysis model."""
        try:
            from insightface.app import FaceAnalysis
        except ImportError:
            raise ImportError(
                "insightface is required. Install with: pip install insightface"
            )

        providers = (
            ["CUDAExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )

        self._face_analysis = FaceAnalysis(root="./models", providers=providers)
        self._face_analysis.prepare(
            ctx_id=0 if self.device == "cuda" else -1, det_size=self.det_size
        )

    def detect(self, frame: Frame) -> List[FaceBBox]:
        """Detect faces in a frame."""
        faces, _ = self._detect_full(frame)
        return faces

    def _detect_full(self, frame: Frame) -> Tuple[List[FaceBBox], list]:
        """Internal method to detect faces and get full insightface results."""
        if self._face_analysis is None:
            self.load_model()

        if frame.shape[2] == 3:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame_rgb = frame

        face_results = self._face_analysis.get(frame_rgb)

        bboxes = []
        valid_faces = []
        for face in face_results:
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
            bboxes.append(face_bbox)
            valid_faces.append(face)

        return bboxes, valid_faces

    def detect_and_embed(self, frame: Frame) -> List[Tuple[FaceBBox, Embedding]]:
        """
        Detect faces and extract embeddings in one pass.

        Args:
            frame: Input image/frame

        Returns:
            List of tuples (FaceBBox, Embedding)
        """
        bboxes, faces = self._detect_full(frame)

        results = []
        for bbox, face in zip(bboxes, faces):
            if hasattr(face, "embedding") and face.embedding is not None:
                embedding = Embedding(
                    vector=face.embedding,
                    model_name="insightface_arcface",
                    normalized=True,
                )
                results.append((bbox, embedding))

        return results

    def detect_full(self, frame: Frame) -> List[Tuple[FaceBBox, Landmarks, Embedding]]:
        """
        Get complete face information: bbox, landmarks, and embedding.

        Args:
            frame: Input image/frame

        Returns:
            List of tuples (FaceBBox, Landmarks, Embedding)
        """
        bboxes, faces = self._detect_full(frame)

        results = []
        for bbox, face in zip(bboxes, faces):
            # Extract landmarks
            landmarks = None
            if hasattr(face, "kps") and face.kps is not None:
                landmarks = Landmarks(points=face.kps, confidence=bbox.confidence)

            # Extract embedding
            embedding = None
            if hasattr(face, "embedding") and face.embedding is not None:
                embedding = Embedding(
                    vector=face.embedding,
                    model_name="insightface_arcface",
                    normalized=True,
                )

            results.append((bbox, landmarks, embedding))

        return results
