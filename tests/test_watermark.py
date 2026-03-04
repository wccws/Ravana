"""
Tests for the watermarking module.
"""

import numpy as np
import pytest

from face_swap.watermark.watermarker import (
    InvisibleWatermarker,
    ProvenanceMetadata,
    WatermarkConfig,
)


@pytest.fixture
def watermarker():
    cfg = WatermarkConfig(enabled=True, strength=8.0)
    return InvisibleWatermarker(cfg)


@pytest.fixture
def test_image():
    """720p-ish test image with random content."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


class TestWatermarkConfig:
    def test_defaults(self):
        cfg = WatermarkConfig()
        assert cfg.enabled is False
        assert cfg.strength == 5.0

    def test_disabled_passthrough(self):
        wm = InvisibleWatermarker(WatermarkConfig(enabled=False))
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        result = wm.embed(img)
        assert np.array_equal(result, img)


class TestProvenanceMetadata:
    def test_json_roundtrip(self):
        meta = ProvenanceMetadata(
            timestamp="2026-01-01T00:00:00Z",
            model_name="inswapper",
            model_version="v0.7",
            is_manipulated=True,
        )
        json_str = meta.to_json()
        restored = ProvenanceMetadata.from_json(json_str)
        assert restored.model_name == "inswapper"
        assert restored.is_manipulated is True


class TestInvisibleWatermarker:
    def test_embed_does_not_crash(self, watermarker, test_image):
        result = watermarker.embed(test_image)
        assert result.shape == test_image.shape
        assert result.dtype == test_image.dtype

    def test_embed_is_nearly_invisible(self, watermarker, test_image):
        result = watermarker.embed(test_image)
        diff = np.abs(result.astype(float) - test_image.astype(float))
        # Average pixel change should be very small
        assert diff.mean() < 5.0

    def test_create_provenance(self, watermarker, test_image):
        meta = watermarker.create_provenance(
            source_image=test_image,
            model_name="inswapper",
            model_version="v0.7",
        )
        assert meta.is_manipulated is True
        assert meta.model_name == "inswapper"
        assert len(meta.source_hash) > 0
