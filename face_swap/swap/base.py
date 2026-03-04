"""
Base class for face swap models.

As per PRD Section 5.6, this defines the interface for combining source identity
with target frame features to generate a swapped face.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..core.types import AlignedFace, Embedding, SwapResult


class FaceSwapper(ABC):
    """
    Abstract base class for face swap models.

    Combines source identity embedding with target frame features
    to generate a swapped face preserving target expression, pose, and lighting.
    """

    def __init__(
        self,
        device: str = "cuda",
        resolution: int = 256,
        use_enhancer: bool = False,
    ):
        """
        Initialize the face swapper.

        Args:
            device: Device to run inference on ("cuda" or "cpu")
            resolution: Output resolution (256 or 512)
            use_enhancer: Whether to use GAN-based enhancement
        """
        self.device = device
        self.resolution = resolution
        self.use_enhancer = use_enhancer
        self._model = None

    @abstractmethod
    def swap(
        self,
        target_aligned: AlignedFace,
        source_embedding: Embedding,
    ) -> SwapResult:
        """
        Generate a swapped face.

        Args:
            target_aligned: Aligned target face (provides pose, expression, lighting)
            source_embedding: Source identity embedding

        Returns:
            SwapResult with swapped face and mask
        """

    def swap_multi(
        self,
        target_faces: List[AlignedFace],
        source_embedding: Embedding,
    ) -> List[SwapResult]:
        """
        Swap multiple target faces with the same source identity.

        Args:
            target_faces: List of aligned target faces
            source_embedding: Source identity embedding

        Returns:
            List of SwapResult objects
        """
        return [self.swap(face, source_embedding) for face in target_faces]

    @abstractmethod
    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the face swap model.

        Args:
            model_path: Path to model weights (optional)
        """

    @abstractmethod
    def get_mask(self, swapped_face: np.ndarray) -> np.ndarray:
        """
        Generate a blending mask for the swapped face.

        Args:
            swapped_face: Swapped face image

        Returns:
            Binary or soft mask
        """
