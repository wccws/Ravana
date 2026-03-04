"""
Unit tests for core data types and utilities.

Validates the typed data structures defined in PRD Section 7.1.
"""

import numpy as np
import pytest

from face_swap.core.types import (
    AlignedFace,
    Embedding,
    FaceBBox,
    Landmarks,
    PipelineResult,
    Point,
    SwapResult,
)

# ------------------------------------------------------------------
# FaceBBox
# ------------------------------------------------------------------


class TestFaceBBox:
    def test_width_height(self):
        bbox = FaceBBox(x1=10, y1=20, x2=110, y2=80, confidence=0.9)
        assert bbox.width == 100
        assert bbox.height == 60

    def test_center(self):
        bbox = FaceBBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        assert bbox.center.x == 50
        assert bbox.center.y == 50

    def test_scale(self):
        bbox = FaceBBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        scaled = bbox.scale(2.0)
        assert scaled.width == 200
        assert scaled.height == 200
        # Center should remain the same
        assert scaled.center.x == 50
        assert scaled.center.y == 50

    def test_to_tuple(self):
        bbox = FaceBBox(x1=1, y1=2, x2=3, y2=4, confidence=0.5)
        assert bbox.to_tuple() == (1, 2, 3, 4)


# ------------------------------------------------------------------
# Landmarks
# ------------------------------------------------------------------


class TestLandmarks:
    def test_from_list(self):
        lm = Landmarks(points=[[1.0, 2.0], [3.0, 4.0]])
        assert lm.num_points == 2
        assert isinstance(lm.points, np.ndarray)

    def test_from_numpy(self):
        arr = np.zeros((68, 2), dtype=np.float32)
        lm = Landmarks(points=arr)
        assert lm.num_points == 68

    def test_eye_centers_68(self):
        # Generate 68 arbitrary points
        pts = np.arange(136, dtype=np.float32).reshape(68, 2)
        lm = Landmarks(points=pts)
        left, right = lm.get_eye_centers()
        # Left eye = mean of points 36..41
        assert isinstance(left, Point)


# ------------------------------------------------------------------
# Embedding
# ------------------------------------------------------------------


class TestEmbedding:
    def test_normalize(self):
        vec = np.array([3.0, 4.0], dtype=np.float32)
        emb = Embedding(vector=vec, model_name="test", normalized=False)
        normed = emb.normalize()
        assert normed.normalized is True
        assert np.isclose(np.linalg.norm(normed.vector), 1.0)

    def test_cosine_similarity_identical(self):
        vec = np.random.randn(512).astype(np.float32)
        e1 = Embedding(vector=vec, model_name="test")
        e2 = Embedding(vector=vec.copy(), model_name="test")
        assert np.isclose(e1.cosine_similarity(e2), 1.0, atol=1e-5)

    def test_cosine_similarity_orthogonal(self):
        e1 = Embedding(vector=np.array([1, 0], dtype=np.float32), model_name="test")
        e2 = Embedding(vector=np.array([0, 1], dtype=np.float32), model_name="test")
        assert np.isclose(e1.cosine_similarity(e2), 0.0, atol=1e-5)

    def test_from_list(self):
        emb = Embedding(vector=[1.0, 2.0, 3.0], model_name="test")
        assert isinstance(emb.vector, np.ndarray)
        assert emb.dimension == 3


# ------------------------------------------------------------------
# AlignedFace
# ------------------------------------------------------------------


class TestAlignedFace:
    def test_shape(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        bbox = FaceBBox(x1=0, y1=0, x2=100, y2=100, confidence=0.9)
        af = AlignedFace(
            image=img,
            transformation_matrix=np.eye(2, 3, dtype=np.float32),
            original_bbox=bbox,
        )
        assert af.shape == (256, 256, 3)


# ------------------------------------------------------------------
# SwapResult / PipelineResult
# ------------------------------------------------------------------


class TestSwapResult:
    def test_shape(self):
        face = np.zeros((128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        emb = Embedding(vector=np.zeros(512), model_name="test")
        bbox = FaceBBox(x1=0, y1=0, x2=128, y2=128, confidence=0.9)
        aligned = AlignedFace(
            image=face,
            transformation_matrix=np.eye(2, 3, dtype=np.float32),
            original_bbox=bbox,
        )
        result = SwapResult(
            swapped_face=face,
            mask=mask,
            source_embedding=emb,
            target_aligned=aligned,
        )
        assert result.shape == (128, 128, 3)


class TestPipelineResult:
    def test_basic(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        pr = PipelineResult(
            output_frame=frame, swap_results=[], processing_time_ms=15.3
        )
        assert pr.processing_time_ms == 15.3
