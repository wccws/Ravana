"""
Model versioning and management.

As per PRD Section 8.3:
- Pre-trained model weights should be versioned and downloadable separately.
- The SDK must expose a mechanism to load different model versions
  (e.g., fast vs. high-quality) and roll back to previous versions
  if regressions are detected.
"""

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("face_swap.models")


@dataclass
class ModelInfo:
    """Metadata for a single model weight file."""

    name: str
    version: str
    path: str
    format: str  # "onnx", "pth", "pt"
    resolution: int = 128
    description: str = ""
    sha256: str = ""
    download_url: str = ""

    @property
    def is_downloaded(self) -> bool:
        return os.path.exists(self.path)


@dataclass
class ModelRegistry:
    """Registry of all known models with their versions."""

    models: Dict[str, List[ModelInfo]] = field(default_factory=dict)

    def register(self, model: ModelInfo) -> None:
        """Register a model (or a new version of an existing model)."""
        key = model.name
        if key not in self.models:
            self.models[key] = []
        # Avoid duplicate version entries
        for existing in self.models[key]:
            if existing.version == model.version:
                return
        self.models[key].append(model)
        self.models[key].sort(key=lambda m: m.version, reverse=True)

    def get_latest(self, name: str) -> Optional[ModelInfo]:
        """Return the latest version of a model."""
        versions = self.models.get(name, [])
        return versions[0] if versions else None

    def get_version(self, name: str, version: str) -> Optional[ModelInfo]:
        """Return a specific version of a model."""
        for m in self.models.get(name, []):
            if m.version == version:
                return m
        return None

    def list_versions(self, name: str) -> List[str]:
        """List all registered versions for a model."""
        return [m.version for m in self.models.get(name, [])]

    def list_models(self) -> List[str]:
        """List all registered model names."""
        return list(self.models.keys())


class ModelManager:
    """
    High-level manager for model discovery, download, verification,
    and version rollback.

    Usage:
        >>> mgr = ModelManager("./models")
        >>> mgr.ensure_model("inswapper", version="v0.7")
        >>> info = mgr.get_model("inswapper")
    """

    MANIFEST_FILE = "manifest.json"

    # Built-in model catalogue (can be extended at runtime)
    _DEFAULT_MODELS: List[ModelInfo] = [
        ModelInfo(
            name="inswapper",
            version="v0.7",
            path="inswapper_128.onnx",
            format="onnx",
            resolution=128,
            description="InsightFace InSwapper 128×128 (fast)",
            download_url=(
                "https://github.com/deepinsight/insightface/releases"
                "/download/v0.7/inswapper_128.onnx"
            ),
        ),
        ModelInfo(
            name="simswap_256",
            version="v1.0",
            path="simswap_256.onnx",
            format="onnx",
            resolution=256,
            description="SimSwap 256×256 (balanced)",
        ),
        ModelInfo(
            name="simswap_512",
            version="v1.0",
            path="simswap_512.onnx",
            format="onnx",
            resolution=512,
            description="SimSwap 512×512 (best quality)",
        ),
    ]

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.registry = ModelRegistry()
        self._active_versions: Dict[str, str] = {}

        # Seed the registry with defaults
        for m in self._DEFAULT_MODELS:
            full = ModelInfo(
                **{
                    **m.__dict__,
                    "path": str(self.models_dir / m.path),
                }
            )
            self.registry.register(full)

        # Load persisted manifest on top (adds user-registered models)
        self._load_manifest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model(
        self, name: str, version: Optional[str] = None
    ) -> Optional[ModelInfo]:
        """
        Get model info, using the active version or the latest.

        Args:
            name:    Model name (e.g. ``inswapper``).
            version: Specific version string, or ``None`` for latest/active.

        Returns:
            ``ModelInfo`` or ``None``.
        """
        if version:
            return self.registry.get_version(name, version)
        if name in self._active_versions:
            return self.registry.get_version(name, self._active_versions[name])
        return self.registry.get_latest(name)

    def set_active_version(self, name: str, version: str) -> None:
        """Pin * name* to a specific *version* (survives restarts via manifest)."""
        info = self.registry.get_version(name, version)
        if info is None:
            raise ValueError(f"Model {name} version {version} not registered.")
        self._active_versions[name] = version
        self._save_manifest()
        logger.info("Pinned %s to version %s", name, version)

    def rollback(self, name: str) -> Optional[ModelInfo]:
        """Roll back to the previous version of *name*.

        Returns the ``ModelInfo`` that is now active, or ``None`` if
        there is nothing to roll back to.
        """
        versions = self.registry.list_versions(name)
        if len(versions) < 2:
            logger.warning(
                "Cannot rollback %s — only %d version(s).", name, len(versions)
            )
            return None

        current = self._active_versions.get(name, versions[0])
        try:
            idx = versions.index(current)
        except ValueError:
            idx = 0
        prev = versions[min(idx + 1, len(versions) - 1)]
        self.set_active_version(name, prev)
        return self.registry.get_version(name, prev)

    def ensure_model(self, name: str, version: Optional[str] = None) -> ModelInfo:
        """
        Make sure the model is downloaded and verified.

        Downloads if necessary, verifies the SHA-256 checksum, and
        returns the ``ModelInfo`` with a valid local path.
        """
        info = self.get_model(name, version)
        if info is None:
            raise ValueError(f"Unknown model: {name} (version={version})")

        if info.is_downloaded:
            if info.sha256 and not self._verify_sha256(info.path, info.sha256):
                logger.warning("SHA-256 mismatch for %s — re-downloading.", info.path)
                os.remove(info.path)
            else:
                return info

        if not info.download_url:
            raise FileNotFoundError(
                f"Model not found locally and no download URL: {info.path}"
            )

        self._download(info.download_url, info.path)
        if info.sha256 and not self._verify_sha256(info.path, info.sha256):
            raise RuntimeError(f"Downloaded model failed checksum: {info.path}")

        return info

    def register_model(self, model: ModelInfo) -> None:
        """Register a user-supplied model (persisted in manifest)."""
        self.registry.register(model)
        self._save_manifest()

    def list_models(self) -> List[str]:
        return self.registry.list_models()

    def list_versions(self, name: str) -> List[str]:
        return self.registry.list_versions(name)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _manifest_path(self) -> Path:
        return self.models_dir / self.MANIFEST_FILE

    def _save_manifest(self) -> None:
        data = {
            "active_versions": self._active_versions,
            "user_models": [
                m.__dict__
                for name in self.registry.models
                for m in self.registry.models[name]
                if m.name not in {d.name for d in self._DEFAULT_MODELS}
                or m.version
                not in {d.version for d in self._DEFAULT_MODELS if d.name == m.name}
            ],
        }
        with open(self._manifest_path(), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    def _load_manifest(self) -> None:
        path = self._manifest_path()
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._active_versions = data.get("active_versions", {})
            for md in data.get("user_models", []):
                self.registry.register(ModelInfo(**md))
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            logger.warning("Corrupt manifest ignored: %s", exc)

    # ------------------------------------------------------------------
    # Download / Verify
    # ------------------------------------------------------------------

    @staticmethod
    def _download(url: str, dest: str) -> None:
        import urllib.request

        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        logger.info("Downloading %s → %s …", url, dest)
        urllib.request.urlretrieve(url, dest)
        logger.info("Download complete.")

    @staticmethod
    def _verify_sha256(path: str, expected: str) -> bool:
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1 << 20)
                if not chunk:
                    break
                sha.update(chunk)
        return sha.hexdigest() == expected
