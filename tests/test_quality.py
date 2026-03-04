"""
Tests for the quality validator module.
"""

import numpy as np
import pytest

from face_swap.core.quality import QualityCode, QualityValidator
from face_swap.core.types import AlignedFace, Embedding, FaceBBox, SwapResult


@pytest.fixture
def validator():
    return QualityValidator()


@pytest.fixture
def good_bbox():
    return FaceBBox(x1=100, y1=100, x2=300, y2=300, confidence=0.95)


@pytest.fixture
def small_bbox():
    return FaceBBox(x1=100, y1=100, x2=115, y2=115, confidence=0.95)


@pytest.fixture
def low_confidence_bbox():
    return FaceBBox(x1=100, y1=100, x2=300, y2=300, confidence=0.2)


# ------------------------------------------------------------------
# Detection validation
# ------------------------------------------------------------------


class TestDetectionValidation:
    def test_valid_detection(self, validator, good_bbox):
        report = validator.validate_detection(good_bbox, (720, 1280))
        assert report.passed

    def test_low_confidence(self, validator, low_confidence_bbox):
        report = validator.validate_detection(low_confidence_bbox, (720, 1280))
        assert report.code == QualityCode.LOW_DETECTION_CONFIDENCE

    def test_face_too_small(self, validator, small_bbox):
        report = validator.validate_detection(small_bbox, (720, 1280))
        assert report.code == QualityCode.FACE_TOO_SMALL

    def test_face_out_of_bounds(self, validator):
        oob_bbox = FaceBBox(x1=-200, y1=-200, x2=-50, y2=-50, confidence=0.9)
        report = validator.validate_detection(oob_bbox, (720, 1280))
        assert report.code == QualityCode.FACE_OUT_OF_BOUNDS


# ------------------------------------------------------------------
# Swap validation
# ------------------------------------------------------------------


class TestSwapValidation:
    def _make_swap_result(self, face_img=None, quality_score=0.85):
        if face_img is None:
            face_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        mask = np.ones((128, 128), dtype=np.float32)
        emb = Embedding(vector=np.zeros(512), model_name="test")
        bbox = FaceBBox(x1=0, y1=0, x2=128, y2=128, confidence=0.9)
        aligned = AlignedFace(
            image=face_img,
            transformation_matrix=np.eye(2, 3, dtype=np.float32),
            original_bbox=bbox,
        )
        return SwapResult(
            swapped_face=face_img,
            mask=mask,
            source_embedding=emb,
            target_aligned=aligned,
            quality_score=quality_score,
        )

    def test_valid_swap(self, validator):
        result = self._make_swap_result()
        report = validator.validate_swap(result)
        assert report.passed or report.code in (QualityCode.OK, QualityCode.SWAP_BLURRY)

    def test_blurry_swap(self, validator):
        # Create a completely uniform (blurry) image
        blank = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = self._make_swap_result(face_img=blank)
        report = validator.validate_swap(result)
        assert report.code == QualityCode.SWAP_BLURRY

    def test_low_quality_score(self, validator):
        result = self._make_swap_result(quality_score=0.1)
        report = validator.validate_swap(result)
        # Should fail on either blur or quality
        assert not report.passed or report.score < 0.3

    def test_should_fallback(self, validator):
        blank = np.full((128, 128, 3), 128, dtype=np.uint8)
        result = self._make_swap_result(face_img=blank)
        report = validator.validate_swap(result)
        assert validator.should_fallback(report) is True
