"""
Tests for the model manager module.
"""

from pathlib import Path

import pytest

from face_swap.core.model_manager import ModelInfo, ModelManager, ModelRegistry


@pytest.fixture
def tmp_models_dir(tmp_path):
    return str(tmp_path / "models")


@pytest.fixture
def manager(tmp_models_dir):
    return ModelManager(models_dir=tmp_models_dir)


class TestModelRegistry:
    def test_register_and_get(self):
        reg = ModelRegistry()
        info = ModelInfo(
            name="test_model", version="v1.0", path="/tmp/test.onnx", format="onnx"
        )
        reg.register(info)
        assert reg.get_latest("test_model") == info

    def test_multiple_versions(self):
        reg = ModelRegistry()
        v1 = ModelInfo(name="m", version="v1.0", path="/tmp/v1.onnx", format="onnx")
        v2 = ModelInfo(name="m", version="v2.0", path="/tmp/v2.onnx", format="onnx")
        reg.register(v1)
        reg.register(v2)
        assert reg.get_latest("m").version == "v2.0"
        assert len(reg.list_versions("m")) == 2

    def test_no_duplicates(self):
        reg = ModelRegistry()
        info = ModelInfo(name="m", version="v1.0", path="/tmp/m.onnx", format="onnx")
        reg.register(info)
        reg.register(info)
        assert len(reg.list_versions("m")) == 1


class TestModelManager:
    def test_default_models_registered(self, manager):
        models = manager.list_models()
        assert "inswapper" in models

    def test_get_model(self, manager):
        info = manager.get_model("inswapper")
        assert info is not None
        assert info.name == "inswapper"

    def test_unknown_model(self, manager):
        info = manager.get_model("nonexistent")
        assert info is None

    def test_set_active_version(self, manager):
        manager.set_active_version("inswapper", "v0.7")
        info = manager.get_model("inswapper")
        assert info.version == "v0.7"

    def test_register_custom_model(self, manager, tmp_models_dir):
        custom = ModelInfo(
            name="custom_swap",
            version="v0.1",
            path=str(Path(tmp_models_dir) / "custom.onnx"),
            format="onnx",
            description="Test custom model",
        )
        manager.register_model(custom)
        assert "custom_swap" in manager.list_models()

    def test_manifest_persistence(self, tmp_models_dir):
        mgr1 = ModelManager(models_dir=tmp_models_dir)
        mgr1.set_active_version("inswapper", "v0.7")

        mgr2 = ModelManager(models_dir=tmp_models_dir)
        info = mgr2.get_model("inswapper")
        assert info.version == "v0.7"

    def test_rollback_no_previous(self, manager):
        result = manager.rollback("inswapper")
        # Only one version registered by default, so rollback gives same
        assert result is None or result.version == "v0.7"
