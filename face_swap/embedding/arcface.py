"""
ArcFace embedding extractor using InsightFace.

As per PRD Section 5.5, this uses ArcFace-style encoder for identity embeddings.
"""

from typing import Optional

import cv2
import numpy as np

from ..core.types import AlignedFace, Embedding
from .base import IdentityEmbedder


class ArcFaceEmbedder(IdentityEmbedder):
    """
    ArcFace identity embedding extractor.

    Extracts 512-dimensional identity embeddings that are
    relatively invariant to expression and lighting.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = "buffalo_l",  # InsightFace model name
        embedding_dim: int = 512,
    ):
        """
        Initialize ArcFace embedder.

        Args:
            device: Device to run inference on
            model_name: InsightFace model name
            embedding_dim: Output embedding dimension (512 for ArcFace)
        """
        super().__init__(device, embedding_dim)
        self.model_name = model_name
        self._face_analysis = None

    def load_model(self) -> None:
        """Load the ArcFace model using InsightFace."""
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

        self._face_analysis = FaceAnalysis(
            name=self.model_name, root="./models", providers=providers
        )
        self._face_analysis.prepare(
            ctx_id=0 if self.device == "cuda" else -1,
            det_size=(112, 112),  # ArcFace expects 112x112
        )

    def extract(self, aligned_face: AlignedFace) -> Embedding:
        """
        Extract identity embedding from an aligned face.

        Args:
            aligned_face: Cropped and aligned face image

        Returns:
            Embedding vector
        """
        if self._face_analysis is None:
            self.load_model()

        # Get the face image
        face_img = aligned_face.image

        # ArcFace expects 112x112 RGB image
        if face_img.shape[:2] != (112, 112):
            face_img = cv2.resize(face_img, (112, 112))

        # Convert BGR to RGB if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            face_rgb = face_img

        # Get embedding using recognition model directly
        # InsightFace expects a batch of images
        embedding = self._face_analysis.models["recognition"].get(face_rgb)

        return Embedding(
            vector=embedding, model_name=f"arcface_{self.model_name}", normalized=True
        )

    def extract_from_image(
        self, image: np.ndarray, bbox: Optional[tuple] = None
    ) -> Embedding:
        """
        Extract embedding directly from an image.

        Args:
            image: Input image (BGR format)
            bbox: Optional bounding box (x1, y1, x2, y2)

        Returns:
            Embedding vector
        """
        if self._face_analysis is None:
            self.load_model()

        # Convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image

        if bbox is not None:
            # Crop to bbox
            x1, y1, x2, y2 = map(int, bbox)
            image_rgb = image_rgb[y1:y2, x1:x2]

        # Resize to 112x112 if needed
        if image_rgb.shape[:2] != (112, 112):
            image_rgb = cv2.resize(image_rgb, (112, 112))

        # Get embedding
        embedding = self._face_analysis.models["recognition"].get(image_rgb)

        return Embedding(
            vector=embedding, model_name=f"arcface_{self.model_name}", normalized=True
        )
