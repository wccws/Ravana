"""
Tests for the config loader module.
"""

import pytest
import yaml

from face_swap.api import FaceSwapConfig
from face_swap.core.config_loader import (
    load_config,
    load_face_swap_config,
    load_pipeline_config,
)
from face_swap.pipeline import PipelineConfig


@pytest.fixture
def sample_yaml(tmp_path):
    """Create a temporary YAML config file."""
    config = {
        "device": "cpu",
        "detection": {
            "model": "retinaface",
            "confidence_threshold": 0.6,
        },
        "alignment": {"crop_size": 512},
        "swap": {
            "model": "simswap",
            "model_path": "./models/simswap_512.onnx",
        },
        "blending": {
            "mode": "poisson",
            "color_correction": False,
        },
        "temporal": {
            "enabled": False,
            "smooth_factor": 0.5,
        },
        "performance": {"batch_size": 4},
        "quality_presets": {
            "low": {"crop_size": 128, "blend_mode": "alpha"},
            "high": {"crop_size": 512, "blend_mode": "feather"},
        },
    }
    path = tmp_path / "test_config.yaml"
    with open(path, "w") as f:
        yaml.dump(config, f)
    return path


class TestLoadConfig:
    def test_load_from_file(self, sample_yaml):
        data = load_config(sample_yaml)
        assert data["device"] == "cpu"
        assert data["detection"]["confidence_threshold"] == 0.6

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/config.yaml")


class TestLoadPipelineConfig:
    def test_basic(self, sample_yaml):
        cfg = load_pipeline_config(sample_yaml)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.device == "cpu"
        assert cfg.crop_size == 512
        assert cfg.swap_model == "simswap"
        assert cfg.blend_mode == "poisson"
        assert cfg.color_correction is False

    def test_overrides(self, sample_yaml):
        cfg = load_pipeline_config(sample_yaml, overrides={"device": "cuda"})
        assert cfg.device == "cuda"


class TestLoadFaceSwapConfig:
    def test_basic(self, sample_yaml):
        cfg = load_face_swap_config(sample_yaml)
        assert isinstance(cfg, FaceSwapConfig)
        assert cfg.device == "cpu"

    def test_quality_preset(self, sample_yaml):
        cfg = load_face_swap_config(sample_yaml, quality="low")
        assert cfg.quality == "low"
