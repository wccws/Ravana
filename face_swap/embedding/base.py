"""
Base class for identity embedding extractors.

As per PRD Section 5.5, this extracts fixed-length embedding vectors
that capture identity but are invariant to expression and lighting.
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np

from ..core.types import AlignedFace, Embedding


class IdentityEmbedder(ABC):
    """
    Abstract base class for identity embedding extractors.
    """

    def __init__(self, device: str = "cuda", embedding_dim: int = 512):
        """
        Initialize the embedder.

        Args:
            device: Device to run inference on ("cuda" or "cpu")
            embedding_dim: Dimension of output embedding vector
        """
        self.device = device
        self.embedding_dim = embedding_dim
        self._model = None

    @abstractmethod
    def extract(self, aligned_face: AlignedFace) -> Embedding:
        """
        Extract identity embedding from an aligned face.

        Args:
            aligned_face: Cropped and aligned face

        Returns:
            Embedding vector
        """

    def extract_multi(self, aligned_faces: List[AlignedFace]) -> List[Embedding]:
        """
        Extract embeddings from multiple faces.

        Args:
            aligned_faces: List of aligned faces

        Returns:
            List of embeddings
        """
        return [self.extract(face) for face in aligned_faces]

    def extract_average(self, aligned_faces: List[AlignedFace]) -> Embedding:
        """
        Extract and average embeddings from multiple faces.

        As per PRD Section 5.5, this averages embeddings across multiple
        source images when provided for improved robustness.

        Args:
            aligned_faces: List of aligned faces from the same identity

        Returns:
            Averaged embedding vector
        """
        if not aligned_faces:
            raise ValueError("No faces provided for averaging")

        embeddings = self.extract_multi(aligned_faces)

        # Average the vectors
        vectors = np.stack([emb.vector for emb in embeddings])
        avg_vector = np.mean(vectors, axis=0)

        return Embedding(
            vector=avg_vector, model_name=embeddings[0].model_name, normalized=False
        ).normalize()

    @abstractmethod
    def load_model(self) -> None:
        """Load the embedding model."""
