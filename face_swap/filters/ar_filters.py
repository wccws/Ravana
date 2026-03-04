"""
AR-style filter experience.

As per PRD Section 3.2 Use Case 4:
  - Integration with camera apps to provide fun filters
    (celebrity, character, or avatar faces) in real time.

This module provides:
  - Filter presets with named source identities.
  - Overlay effects (frames, stickers, background blur).
  - AR-style real-time filter loop with hot-swappable presets.
  - Shared filter gallery management.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("face_swap.filters")


class OverlayMode(Enum):
    """Types of visual overlay that can be composited on top of a swap."""

    NONE = "none"
    FRAME = "frame"  # Decorative frame around the output
    STICKER = "sticker"  # 2D sticker overlay on the face
    BACKGROUND_BLUR = "bg_blur"
    BACKGROUND_REPLACE = "bg_replace"
    COLOR_GRADE = "color_grade"


@dataclass
class FilterPreset:
    """A named filter combining a source identity with overlays.

    Attributes:
        name:            Human-readable filter name (e.g., "Celebrity A").
        source_images:   Paths to source identity image(s).
        thumbnail:       Path to thumbnail image for UI.
        overlay_mode:    Type of visual overlay.
        overlay_asset:   Path to overlay image asset (if applicable).
        background:      Path to background image (for BG replace).
        color_lut:       Path to 3D LUT for colour grading.
        swap_quality:    Quality preset for this filter.
        description:     Filter description.
        tags:            Searchable tags.
    """

    name: str
    source_images: List[str]
    thumbnail: str = ""
    overlay_mode: OverlayMode = OverlayMode.NONE
    overlay_asset: str = ""
    background: str = ""
    color_lut: str = ""
    swap_quality: str = "medium"
    description: str = ""
    tags: List[str] = field(default_factory=list)


class FilterGallery:
    """
    Manages a collection of filter presets.

    Provides CRUD operations and search for AR filter experiences.
    """

    def __init__(self, filters_dir: Optional[str] = None):
        self._filters: Dict[str, FilterPreset] = {}
        self._filters_dir = Path(filters_dir) if filters_dir else None

    def add(self, preset: FilterPreset) -> None:
        """Add a filter preset to the gallery."""
        self._filters[preset.name] = preset

    def remove(self, name: str) -> None:
        """Remove a filter preset."""
        self._filters.pop(name, None)

    def get(self, name: str) -> Optional[FilterPreset]:
        """Get a filter preset by name."""
        return self._filters.get(name)

    def list_all(self) -> List[FilterPreset]:
        """List all available filter presets."""
        return list(self._filters.values())

    def search(self, query: str) -> List[FilterPreset]:
        """Search filters by name or tags."""
        query_lower = query.lower()
        return [
            f
            for f in self._filters.values()
            if query_lower in f.name.lower()
            or any(query_lower in t.lower() for t in f.tags)
        ]

    def load_from_directory(self, path: Optional[str] = None) -> int:
        """
        Load filter presets from a directory of JSON definitions.

        Each ``.json`` file defines a ``FilterPreset``.

        Returns:
            Number of filters loaded.
        """
        import json

        scan_dir = Path(path) if path else self._filters_dir
        if scan_dir is None or not scan_dir.exists():
            return 0

        count = 0
        for json_file in scan_dir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                data["overlay_mode"] = OverlayMode(data.get("overlay_mode", "none"))
                preset = FilterPreset(**data)
                self.add(preset)
                count += 1
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("Skipping %s: %s", json_file, e)

        logger.info("Loaded %d filter presets from %s", count, scan_dir)
        return count


class ARFilterEngine:
    """
    Real-time AR filter engine.

    Wraps the face swap pipeline with hot-swappable filter presets,
    overlay compositing, and background effects.

    Usage:
        >>> engine = ARFilterEngine()
        >>> engine.set_filter(FilterPreset(name="Fun", source_images=["celeb.jpg"]))
        >>> engine.start(camera_id=0)
    """

    def __init__(self, device: str = "cuda", quality: str = "medium"):
        from face_swap.pipeline import FaceSwapPipeline

        self._device = device
        self._quality = quality
        self._pipeline: Optional[FaceSwapPipeline] = None
        self._current_filter: Optional[FilterPreset] = None
        self._source_embedding = None
        self._overlay_cache: Dict[str, np.ndarray] = {}
        self._bg_image: Optional[np.ndarray] = None

    def set_filter(self, preset: FilterPreset) -> None:
        """
        Apply a new filter preset.

        Loads the source identity and any overlay assets.
        This can be called at any time — even during a live session.
        """
        from face_swap.pipeline import FaceSwapPipeline, PipelineConfig

        self._current_filter = preset

        # Build pipeline for this filter's quality level
        config = PipelineConfig(
            device=self._device,
            async_detection=True,
            enable_temporal=True,
        )
        self._pipeline = FaceSwapPipeline(config)
        self._pipeline.initialize()

        # Extract source identity
        source_images = []
        for path in preset.source_images:
            img = cv2.imread(path)
            if img is not None:
                source_images.append(img)

        if not source_images:
            raise ValueError(f"No valid source images for filter '{preset.name}'")

        if len(source_images) == 1:
            self._source_embedding = self._pipeline.extract_source_embedding(
                source_images[0]
            )
        else:
            self._source_embedding = self._pipeline.extract_source_embedding_multi(
                source_images
            )

        # Load overlay assets
        if preset.overlay_asset and preset.overlay_mode in (
            OverlayMode.FRAME,
            OverlayMode.STICKER,
        ):
            overlay = cv2.imread(preset.overlay_asset, cv2.IMREAD_UNCHANGED)
            if overlay is not None:
                self._overlay_cache[preset.name] = overlay

        # Load background
        if preset.background and preset.overlay_mode == OverlayMode.BACKGROUND_REPLACE:
            bg = cv2.imread(preset.background)
            if bg is not None:
                self._bg_image = bg

        logger.info("Filter applied: %s", preset.name)

    def process_frame(self, frame: np.ndarray, frame_idx: int = 0) -> np.ndarray:
        """
        Process a single camera frame through the current filter.

        Args:
            frame:     BGR uint8 camera frame.
            frame_idx: Sequential frame number.

        Returns:
            Processed frame with face swap and overlay effects.
        """
        if self._pipeline is None or self._source_embedding is None:
            return frame

        # 1. Face swap
        result = self._pipeline.process_video_frame(
            frame, self._source_embedding, frame_idx
        )

        # 2. Apply overlay effects
        if self._current_filter:
            result = self._apply_overlay(result, frame)

        return result

    def start(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        on_frame: Optional[Callable[[np.ndarray, int], None]] = None,
    ) -> None:
        """
        Start the live AR filter experience.

        Args:
            camera_id:  Camera device index.
            resolution: Capture resolution.
            on_frame:   Optional callback receiving (output_frame, frame_idx).
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        window = "Face Swap — AR Filter"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        import time

        frame_idx = 0
        fps_history: List[float] = []

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.time()
                output = self.process_frame(frame, frame_idx)
                elapsed = time.time() - t0

                # FPS overlay
                fps = 1.0 / elapsed if elapsed > 0 else 0
                fps_history.append(fps)
                if len(fps_history) > 30:
                    fps_history.pop(0)
                avg_fps = sum(fps_history) / len(fps_history)

                cv2.putText(
                    output,
                    f"FPS: {avg_fps:.0f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

                if self._current_filter:
                    cv2.putText(
                        output,
                        self._current_filter.name,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )

                cv2.imshow(window, output)

                if on_frame:
                    on_frame(output, frame_idx)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

                frame_idx += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self._pipeline:
                self._pipeline.cleanup()

    # ── Overlay compositing ──────────────────────────────────────────────

    def _apply_overlay(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Apply overlay effects based on the current filter preset."""
        if self._current_filter is None:
            return result

        mode = self._current_filter.overlay_mode

        if mode == OverlayMode.FRAME:
            return self._apply_frame_overlay(result)
        elif mode == OverlayMode.STICKER:
            return self._apply_sticker_overlay(result)
        elif mode == OverlayMode.BACKGROUND_BLUR:
            return self._apply_bg_blur(result, original)
        elif mode == OverlayMode.BACKGROUND_REPLACE:
            return self._apply_bg_replace(result, original)
        elif mode == OverlayMode.COLOR_GRADE:
            return self._apply_color_grade(result)

        return result

    def _apply_frame_overlay(self, result: np.ndarray) -> np.ndarray:
        """Composite a decorative frame overlay."""
        overlay = self._overlay_cache.get(self._current_filter.name)
        if overlay is None:
            return result

        h, w = result.shape[:2]
        overlay_resized = cv2.resize(overlay, (w, h))

        if overlay_resized.shape[2] == 4:
            alpha = overlay_resized[:, :, 3:4].astype(np.float32) / 255.0
            rgb = overlay_resized[:, :, :3]
            result = (rgb * alpha + result * (1 - alpha)).astype(np.uint8)

        return result

    def _apply_sticker_overlay(self, result: np.ndarray) -> np.ndarray:
        """Place a sticker overlay on the face region."""
        # Simplified: overlay centered sticker
        sticker = self._overlay_cache.get(self._current_filter.name)
        if sticker is None:
            return result

        h, w = result.shape[:2]
        sh, sw = sticker.shape[:2]

        # Center
        x = (w - sw) // 2
        y = (h - sh) // 2

        if x < 0 or y < 0:
            scale = min(w / sw, h / sh) * 0.5
            sticker = cv2.resize(sticker, None, fx=scale, fy=scale)
            sh, sw = sticker.shape[:2]
            x = (w - sw) // 2
            y = (h - sh) // 2

        if sticker.shape[2] == 4:
            alpha = sticker[:, :, 3:4].astype(np.float32) / 255.0
            roi = result[y : y + sh, x : x + sw]
            blended = (sticker[:, :, :3] * alpha + roi * (1 - alpha)).astype(np.uint8)
            result[y : y + sh, x : x + sw] = blended

        return result

    def _apply_bg_blur(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Blur the background while keeping the face sharp."""
        blurred = cv2.GaussianBlur(result, (0, 0), 25)

        # Simple face region mask (detection-based)
        if self._pipeline and self._pipeline.detector:
            bboxes = self._pipeline.detector.detect(original)
            mask = np.zeros(result.shape[:2], dtype=np.float32)
            for bbox in bboxes:
                # Expand bbox for more natural look
                cx, cy = int(bbox.center.x), int(bbox.center.y)
                rw, rh = int(bbox.width * 0.8), int(bbox.height * 0.8)
                cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)

            mask = cv2.GaussianBlur(mask, (51, 51), 20)
            mask = mask[:, :, np.newaxis]
            result = (result * mask + blurred * (1 - mask)).astype(np.uint8)

        return result

    def _apply_bg_replace(self, result: np.ndarray, original: np.ndarray) -> np.ndarray:
        """Replace background with a custom image."""
        if self._bg_image is None:
            return result

        h, w = result.shape[:2]
        bg = cv2.resize(self._bg_image, (w, h))

        # Simple face mask
        if self._pipeline and self._pipeline.detector:
            bboxes = self._pipeline.detector.detect(original)
            mask = np.zeros((h, w), dtype=np.float32)
            for bbox in bboxes:
                cx, cy = int(bbox.center.x), int(bbox.center.y)
                rw, rh = int(bbox.width * 1.2), int(bbox.height * 1.5)
                cv2.ellipse(mask, (cx, cy), (rw, rh), 0, 0, 360, 1.0, -1)

            mask = cv2.GaussianBlur(mask, (51, 51), 20)
            mask = mask[:, :, np.newaxis]
            result = (result * mask + bg * (1 - mask)).astype(np.uint8)

        return result

    def _apply_color_grade(self, result: np.ndarray) -> np.ndarray:
        """Apply a colour grading LUT."""
        lut_path = self._current_filter.color_lut
        if not lut_path:
            return result

        # Simple contrast/brightness adjustment as fallback
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        enhanced = cv2.merge([l_channel, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
