"""
Face enhancement / super-resolution module.

As per PRD Section 5.6, provides optional GAN-based refiners and
super-resolution enhancers for post-processing swapped faces.
"""

from .enhancer import (
    CodeFormerEnhancer,
    EnhancementConfig,
    FaceEnhancer,
    GFPGANEnhancer,
    RealESRGANEnhancer,
    create_enhancer,
)

__all__ = [
    "FaceEnhancer",
    "EnhancementConfig",
    "GFPGANEnhancer",
    "RealESRGANEnhancer",
    "CodeFormerEnhancer",
    "create_enhancer",
]
