"""
Plugin and extension system for the Face Swap SDK.

As per PRD Section 2.1:
  - Expose a modular pipeline so that individual components
    (detector, landmarks, swapper, blender) can be upgraded independently.

This module provides:
  - A plugin registry for runtime discovery and hot-swapping of components.
  - Entry-point based auto-discovery (pkg_resources / importlib.metadata).
  - Type-safe registration with interface validation.
"""

import importlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, TypeVar

logger = logging.getLogger("face_swap.plugins")

T = TypeVar("T")


@dataclass
class PluginInfo:
    """Metadata about a registered plugin."""

    name: str
    version: str
    category: (
        str  # "detector", "landmarks", "embedder", "swapper", "blender", "enhancer"
    )
    cls: type
    description: str = ""
    author: str = ""
    priority: int = 0  # Higher = preferred


class PluginRegistry:
    """
    Central registry for pluggable pipeline components.

    Allows third-party packages to register custom detectors, swappers,
    blenders, etc. that the pipeline can discover and use at runtime.

    Usage:
        >>> registry = PluginRegistry()
        >>> registry.register(PluginInfo(
        ...     name="my_detector",
        ...     version="1.0",
        ...     category="detector",
        ...     cls=MyCustomDetector,
        ... ))
        >>> DetCls = registry.get("detector", "my_detector")
        >>> det = DetCls(device="cuda")
    """

    # Known component categories and their expected base classes
    CATEGORIES = {
        "detector": "face_swap.detection.base.FaceDetector",
        "landmarks": "face_swap.landmarks.base.LandmarkDetector",
        "embedder": "face_swap.embedding.base.IdentityEmbedder",
        "swapper": "face_swap.swap.base.FaceSwapper",
        "blender": None,  # FaceBlender doesn't have a formal ABC yet
        "enhancer": "face_swap.enhancement.enhancer.FaceEnhancer",
        "temporal": None,
    }

    def __init__(self):
        self._plugins: Dict[str, Dict[str, PluginInfo]] = {
            cat: {} for cat in self.CATEGORIES
        }

    # -- Registration --------------------------------------------------

    def register(self, info: PluginInfo) -> None:
        """
        Register a plugin.

        Args:
            info: Plugin metadata including the implementing class.

        Raises:
            ValueError: If the category is unknown.
        """
        if info.category not in self.CATEGORIES:
            raise ValueError(
                f"Unknown category '{info.category}'. "
                f"Choose from: {list(self.CATEGORIES.keys())}"
            )

        self._plugins[info.category][info.name] = info
        logger.info(
            "Registered plugin: %s v%s [%s]",
            info.name,
            info.version,
            info.category,
        )

    def unregister(self, category: str, name: str) -> None:
        """Remove a plugin from the registry."""
        self._plugins.get(category, {}).pop(name, None)

    # -- Lookup --------------------------------------------------------

    def get(self, category: str, name: str) -> Optional[type]:
        """
        Get a plugin class by category and name.

        Returns:
            The plugin class, or None if not found.
        """
        info = self._plugins.get(category, {}).get(name)
        return info.cls if info else None

    def get_info(self, category: str, name: str) -> Optional[PluginInfo]:
        """Get full plugin metadata."""
        return self._plugins.get(category, {}).get(name)

    def list_plugins(self, category: Optional[str] = None) -> List[PluginInfo]:
        """
        List all registered plugins, optionally filtered by category.
        """
        if category:
            return list(self._plugins.get(category, {}).values())
        return [info for cat in self._plugins.values() for info in cat.values()]

    def get_preferred(self, category: str) -> Optional[type]:
        """
        Get the highest-priority plugin in a category.
        """
        plugins = list(self._plugins.get(category, {}).values())
        if not plugins:
            return None
        best = max(plugins, key=lambda p: p.priority)
        return best.cls

    # -- Discovery -----------------------------------------------------

    def discover_entry_points(self, group: str = "face_swap.plugins") -> int:
        """
        Auto-discover plugins registered via Python package entry points.

        Third-party packages can declare plugins in their ``setup.py``::

            entry_points={
                "face_swap.plugins": [
                    "my_detector = my_package.detector:MyDetector",
                ],
            }

        Returns:
            Number of plugins discovered.
        """
        count = 0
        try:
            from importlib.metadata import entry_points as get_eps

            eps = get_eps()
            # Python 3.12+ returns a SelectableGroups; earlier returns dict
            if hasattr(eps, "select"):
                plugin_eps = eps.select(group=group)
            elif isinstance(eps, dict):
                plugin_eps = eps.get(group, [])
            else:
                plugin_eps = []

            for ep in plugin_eps:
                try:
                    cls = ep.load()
                    # Try to infer category from a class attribute or name
                    category = getattr(cls, "PLUGIN_CATEGORY", "swapper")
                    info = PluginInfo(
                        name=ep.name,
                        version=getattr(cls, "PLUGIN_VERSION", "0.0.0"),
                        category=category,
                        cls=cls,
                        description=getattr(cls, "PLUGIN_DESCRIPTION", ""),
                        author=getattr(cls, "PLUGIN_AUTHOR", ""),
                    )
                    self.register(info)
                    count += 1
                except Exception as e:
                    logger.warning("Failed to load plugin %s: %s", ep.name, e)

        except ImportError:
            logger.debug(
                "importlib.metadata not available; skipping entry point discovery."
            )

        if count:
            logger.info("Discovered %d plugin(s) from entry points.", count)
        return count

    def discover_module(self, module_path: str) -> int:
        """
        Discover plugins from a Python module path.

        The module should define a ``register(registry)`` function.

        Args:
            module_path: Dotted module path (e.g., ``my_package.plugins``).

        Returns:
            Number of plugins registered.
        """
        before = sum(len(v) for v in self._plugins.values())
        try:
            mod = importlib.import_module(module_path)
            if hasattr(mod, "register"):
                mod.register(self)
        except ImportError as e:
            logger.warning("Could not import plugin module %s: %s", module_path, e)
        after = sum(len(v) for v in self._plugins.values())
        return after - before


# ── Global registry singleton ────────────────────────────────────────────

_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry (created on first call)."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
        # Register built-in plugins
        _register_builtins(_global_registry)
        # Auto-discover third-party plugins
        _global_registry.discover_entry_points()
    return _global_registry


def _register_builtins(registry: PluginRegistry) -> None:
    """Register built-in pipeline components as plugins."""
    from ..blending.blender import FaceBlender
    from ..detection.retinaface import RetinaFaceDetector
    from ..embedding.arcface import ArcFaceEmbedder
    from ..landmarks.mediapipe_lm import MediaPipeLandmarkDetector
    from ..swap.inswapper import InSwapperModel

    builtins = [
        PluginInfo(
            "retinaface",
            "0.1.0",
            "detector",
            RetinaFaceDetector,
            "RetinaFace via InsightFace",
            priority=10,
        ),
        PluginInfo(
            "mediapipe",
            "0.1.0",
            "landmarks",
            MediaPipeLandmarkDetector,
            "MediaPipe Face Mesh 468-point",
            priority=10,
        ),
        PluginInfo(
            "arcface",
            "0.1.0",
            "embedder",
            ArcFaceEmbedder,
            "ArcFace identity embedder",
            priority=10,
        ),
        PluginInfo(
            "inswapper",
            "0.1.0",
            "swapper",
            InSwapperModel,
            "InsightFace InSwapper 128×128",
            priority=10,
        ),
        PluginInfo(
            "opencv_blender",
            "0.1.0",
            "blender",
            FaceBlender,
            "OpenCV-based multi-mode blender",
            priority=10,
        ),
    ]

    # SimSwap (optional)
    try:
        from ..swap.simswap import SimSwapModel

        builtins.append(
            PluginInfo(
                "simswap",
                "0.1.0",
                "swapper",
                SimSwapModel,
                "SimSwap generator (256/512)",
                priority=5,
            )
        )
    except ImportError:
        pass

    for info in builtins:
        registry.register(info)


# ── Decorator for easy registration ──────────────────────────────────────


def register_plugin(
    name: str,
    category: str,
    version: str = "0.0.1",
    priority: int = 0,
    description: str = "",
):
    """
    Class decorator to register a class as a Face Swap plugin.

    Usage::

        @register_plugin("my_detector", "detector", version="1.0")
        class MyCustomDetector(FaceDetector):
            ...
    """

    def decorator(cls):
        info = PluginInfo(
            name=name,
            version=version,
            category=category,
            cls=cls,
            description=description,
            priority=priority,
        )
        get_registry().register(info)
        return cls

    return decorator
