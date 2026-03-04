"""
Multi-model routing and selection.

As per PRD Section 10.3:
  - Support multiple specialized models (e.g., portrait-only vs. wide-angle).
  - Auto-select the best model based on input characteristics.

This module provides:
  - Model pool with named profiles (portrait, wide-angle, group, etc.).
  - Automatic model selection based on face count, pose, resolution.
  - Runtime model switching without pipeline restart.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

logger = logging.getLogger("face_swap.routing")


class SceneType(Enum):
    """Detected scene category for model routing."""

    PORTRAIT = "portrait"  # Single face, close-up
    GROUP = "group"  # Multiple faces
    WIDE_ANGLE = "wide_angle"  # Small faces in wide shot
    PROFILE = "profile"  # Side-view faces
    UNKNOWN = "unknown"


@dataclass
class ModelProfile:
    """Profile for a specialized swap model.

    Attributes:
        name:            Model identifier.
        model_type:      ``inswapper`` or ``simswap``.
        model_path:      Path to model weights.
        resolution:      Face crop resolution.
        scene_types:     Scenes this model excels at.
        max_faces:       Maximum faces it can handle efficiently.
        min_face_size:   Minimum face size (pixels) it works well with.
        quality_score:   Base quality rating (0-1).
        speed_score:     Speed rating (0-1, higher = faster).
    """

    name: str
    model_type: str = "inswapper"
    model_path: Optional[str] = None
    resolution: int = 128
    scene_types: List[SceneType] = field(default_factory=lambda: [SceneType.PORTRAIT])
    max_faces: int = 1
    min_face_size: int = 32
    quality_score: float = 0.8
    speed_score: float = 0.8


class ModelRouter:
    """
    Automatically routes input frames to the best available model.

    Selection criteria:
      1. Number of detected faces.
      2. Face size relative to frame.
      3. Estimated head pose (frontal vs. profile).
      4. User preference (speed vs. quality).
    """

    # Default model pool
    DEFAULT_PROFILES = [
        ModelProfile(
            name="inswapper_fast",
            model_type="inswapper",
            resolution=128,
            scene_types=[SceneType.PORTRAIT, SceneType.GROUP],
            max_faces=5,
            min_face_size=32,
            quality_score=0.7,
            speed_score=1.0,
        ),
        ModelProfile(
            name="simswap_balanced",
            model_type="simswap",
            resolution=256,
            scene_types=[SceneType.PORTRAIT, SceneType.GROUP],
            max_faces=3,
            min_face_size=64,
            quality_score=0.85,
            speed_score=0.6,
        ),
        ModelProfile(
            name="simswap_hq",
            model_type="simswap",
            resolution=512,
            scene_types=[SceneType.PORTRAIT],
            max_faces=1,
            min_face_size=128,
            quality_score=1.0,
            speed_score=0.3,
        ),
    ]

    def __init__(
        self,
        profiles: Optional[List[ModelProfile]] = None,
        prefer_quality: bool = False,
    ):
        """
        Args:
            profiles:       List of model profiles to choose from.
            prefer_quality: If True, prefer quality over speed.
        """
        self._profiles = profiles or list(self.DEFAULT_PROFILES)
        self._prefer_quality = prefer_quality
        self._current_profile: Optional[ModelProfile] = None

    def add_profile(self, profile: ModelProfile) -> None:
        """Add a model profile to the pool."""
        self._profiles.append(profile)

    def remove_profile(self, name: str) -> None:
        """Remove a model profile by name."""
        self._profiles = [p for p in self._profiles if p.name != name]

    def get_profile(self, name: str) -> Optional[ModelProfile]:
        """Get a specific profile by name."""
        for p in self._profiles:
            if p.name == name:
                return p
        return None

    def list_profiles(self) -> List[ModelProfile]:
        """List all available profiles."""
        return list(self._profiles)

    @property
    def current_profile(self) -> Optional[ModelProfile]:
        """The currently selected model profile."""
        return self._current_profile

    # ------------------------------------------------------------------
    # Auto-selection
    # ------------------------------------------------------------------

    def select_model(
        self,
        num_faces: int = 1,
        avg_face_size: float = 128.0,
        frame_shape: Tuple[int, int] = (720, 1280),
        scene_type: SceneType = SceneType.UNKNOWN,
    ) -> ModelProfile:
        """
        Select the best model for the current frame.

        Args:
            num_faces:     Number of detected faces.
            avg_face_size: Average face width in pixels.
            frame_shape:   Frame (H, W).
            scene_type:    Detected scene category.

        Returns:
            Best matching ``ModelProfile``.
        """
        candidates = self._filter_candidates(num_faces, avg_face_size, scene_type)

        if not candidates:
            # Fallback to any profile that can handle the face count
            candidates = [p for p in self._profiles if p.max_faces >= num_faces]
            if not candidates:
                candidates = self._profiles

        # Score each candidate
        scored = []
        for profile in candidates:
            score = self._score_profile(profile, num_faces, avg_face_size, frame_shape)
            scored.append((score, profile))

        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0][1]

        if best != self._current_profile:
            logger.info(
                "Model router: %s → %s (faces=%d, size=%.0f)",
                self._current_profile.name if self._current_profile else "none",
                best.name,
                num_faces,
                avg_face_size,
            )
            self._current_profile = best

        return best

    def classify_scene(
        self,
        num_faces: int,
        avg_face_size: float,
        frame_shape: Tuple[int, int],
        max_yaw: float = 0.0,
    ) -> SceneType:
        """
        Classify the scene type from frame statistics.

        Args:
            num_faces:     Number of faces.
            avg_face_size: Average face width (pixels).
            frame_shape:   Frame (H, W).
            max_yaw:       Maximum absolute yaw angle among faces.

        Returns:
            Detected ``SceneType``.
        """
        h, w = frame_shape
        face_ratio = avg_face_size / max(w, 1)

        if abs(max_yaw) > 30:
            return SceneType.PROFILE

        if num_faces == 1 and face_ratio > 0.15:
            return SceneType.PORTRAIT

        if num_faces > 2:
            return SceneType.GROUP

        if face_ratio < 0.08:
            return SceneType.WIDE_ANGLE

        return SceneType.PORTRAIT

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_candidates(
        self,
        num_faces: int,
        avg_face_size: float,
        scene_type: SceneType,
    ) -> List[ModelProfile]:
        """Filter profiles by hard constraints."""
        return [
            p
            for p in self._profiles
            if p.max_faces >= num_faces
            and p.min_face_size <= avg_face_size
            and (scene_type == SceneType.UNKNOWN or scene_type in p.scene_types)
        ]

    def _score_profile(
        self,
        profile: ModelProfile,
        num_faces: int,
        avg_face_size: float,
        frame_shape: Tuple[int, int],
    ) -> float:
        """Score a profile for the current input (higher = better)."""
        quality_weight = 0.7 if self._prefer_quality else 0.3
        speed_weight = 1.0 - quality_weight

        score = (
            quality_weight * profile.quality_score + speed_weight * profile.speed_score
        )

        # Bonus: resolution matches face size well
        ratio = profile.resolution / max(avg_face_size, 1)
        if 0.5 <= ratio <= 2.0:
            score += 0.1

        # Penalty: model can only handle fewer faces than detected
        if profile.max_faces < num_faces:
            score -= 0.5

        return score
