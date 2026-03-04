"""
Advanced temporal consistency using optical flow and temporal GANs.

As per PRD Section 5.8 (Phase 2 advancement):
  - Use optical flow or face tracking to track face regions across frames.
  - Optionally apply temporal smoothing in latent / embedding space.
  - Use temporal consistency losses or temporal GANs for advanced models.

This module extends the base TemporalSmoother with:
  - Dense optical flow warping (Farneback / RAFT).
  - Latent-space temporal smoothing.
  - Flow-guided blending for ghost-free transitions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger("face_swap.temporal.advanced")


@dataclass
class OpticalFlowConfig:
    """Configuration for optical-flow-based temporal smoothing.

    Attributes:
        method:              ``farneback`` or ``raft`` (GPU-accelerated).
        flow_smooth_factor:  Exponential moving average blend for flow field.
        warp_blend:          Blend ratio between warped previous frame and current.
        max_flow_magnitude:  Discard flow vectors above this (pixel) threshold.
        latent_smoothing:    Whether to smooth in embedding / latent space.
        latent_alpha:        EMA factor for latent-space smoothing.
    """

    method: str = "farneback"
    flow_smooth_factor: float = 0.6
    warp_blend: float = 0.3
    max_flow_magnitude: float = 50.0
    latent_smoothing: bool = True
    latent_alpha: float = 0.7


class OpticalFlowSmoother:
    """
    Advanced temporal smoother using dense optical flow.

    Pipeline:
      1. Compute dense optical flow between consecutive frames.
      2. Warp the previous swapped output using the flow field.
      3. Blend warped previous + current swap to reduce flicker.
      4. (Optional) Smooth embeddings / latent vectors over time.
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None):
        self.config = config or OpticalFlowConfig()

        # History buffers
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_output: Optional[np.ndarray] = None
        self._flow_accum: Optional[np.ndarray] = None

        # Latent-space smoothing state
        self._latent_cache: Dict[int, np.ndarray] = {}

        # Frame counter
        self._frame_count = 0

    def smooth_frame(
        self,
        current_output: np.ndarray,
        current_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Apply flow-guided temporal smoothing to the current swapped output.

        Args:
            current_output: Swapped frame (BGR uint8).
            current_frame:  Original capture frame (for flow computation).

        Returns:
            Temporally smoothed output frame.
        """
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or self._prev_output is None:
            self._prev_gray = gray
            self._prev_output = current_output.copy()
            self._frame_count += 1
            return current_output

        # 1. Compute dense optical flow
        flow = self._compute_flow(self._prev_gray, gray)

        # 2. Accumulate flow with EMA
        if self._flow_accum is None:
            self._flow_accum = flow.copy()
        else:
            alpha = self.config.flow_smooth_factor
            self._flow_accum = alpha * self._flow_accum + (1 - alpha) * flow

        # 3. Clamp extreme flow vectors
        magnitude = np.linalg.norm(self._flow_accum, axis=2)
        mask = magnitude > self.config.max_flow_magnitude
        self._flow_accum[mask] = 0.0

        # 4. Warp previous output using flow
        warped = self._warp_frame(self._prev_output, self._flow_accum)

        # 5. Blend warped previous with current
        blend = self.config.warp_blend
        result = cv2.addWeighted(
            warped.astype(np.float32),
            blend,
            current_output.astype(np.float32),
            1.0 - blend,
            0,
        )
        result = np.clip(result, 0, 255).astype(np.uint8)

        # Update history
        self._prev_gray = gray
        self._prev_output = result.copy()
        self._frame_count += 1

        return result

    def smooth_latent(
        self,
        track_id: int,
        latent: np.ndarray,
    ) -> np.ndarray:
        """
        Smooth a latent / embedding vector over time with EMA.

        Args:
            track_id: Unique face track identifier.
            latent:   Current frame's latent vector.

        Returns:
            Temporally smoothed latent vector.
        """
        if not self.config.latent_smoothing:
            return latent

        alpha = self.config.latent_alpha
        if track_id in self._latent_cache:
            prev = self._latent_cache[track_id]
            smoothed = alpha * prev + (1 - alpha) * latent
        else:
            smoothed = latent

        self._latent_cache[track_id] = smoothed
        return smoothed

    def reset(self) -> None:
        """Clear all temporal state."""
        self._prev_gray = None
        self._prev_output = None
        self._flow_accum = None
        self._latent_cache.clear()
        self._frame_count = 0

    # ── Flow computation ─────────────────────────────────────────────────

    def _compute_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Compute dense optical flow between two grayscale frames."""

        if self.config.method == "farneback":
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            return flow

        elif self.config.method == "raft":
            return self._compute_flow_raft(prev_gray, curr_gray)

        else:
            raise ValueError(f"Unknown flow method: {self.config.method}")

    def _compute_flow_raft(
        self, prev_gray: np.ndarray, curr_gray: np.ndarray
    ) -> np.ndarray:
        """
        Compute flow using RAFT (requires torchvision).

        Falls back to Farneback if RAFT is unavailable.
        """
        try:
            import torch
            from torchvision.models.optical_flow import Raft_Small_Weights, raft_small

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Convert grayscale to 3-channel for RAFT
            prev_rgb = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2RGB)
            curr_rgb = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2RGB)

            # Normalize and convert to tensor
            transforms = Raft_Small_Weights.DEFAULT.transforms()
            prev_t, curr_t = transforms(
                torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0).float(),
                torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0).float(),
            )
            prev_t = prev_t.to(device)
            curr_t = curr_t.to(device)

            model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval()
            with torch.no_grad():
                flow_list = model(prev_t, curr_t)
                flow = flow_list[-1]  # Final flow prediction

            # Convert back to numpy (H, W, 2)
            flow_np = flow[0].permute(1, 2, 0).cpu().numpy()
            return flow_np

        except (ImportError, RuntimeError) as e:
            logger.warning("RAFT unavailable (%s); falling back to Farneback.", e)
            return cv2.calcOpticalFlowFarneback(
                prev_gray,
                curr_gray,
                None,
                0.5,
                3,
                15,
                3,
                5,
                1.2,
                0,
            )

    # ── Warping ──────────────────────────────────────────────────────────

    @staticmethod
    def _warp_frame(frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp a frame using a dense flow field via remap."""
        h, w = flow.shape[:2]

        # Resize flow to match frame if needed
        fh, fw = frame.shape[:2]
        if (h, w) != (fh, fw):
            flow = cv2.resize(flow, (fw, fh))
            flow[:, :, 0] *= fw / w
            flow[:, :, 1] *= fh / h
            h, w = fh, fw

        # Build remap coordinates
        map_y, map_x = np.mgrid[0:h, 0:w].astype(np.float32)
        map_x += flow[:, :, 0]
        map_y += flow[:, :, 1]

        warped = cv2.remap(
            frame,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped


class FlowGuidedBlender:
    """
    Flow-guided blending for ghost-free face transitions.

    When a face appears / disappears or the swap model produces
    a large change between consecutive frames, this blender uses
    the optical flow confidence to decide how aggressively to
    blend the old and new outputs.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        transition_frames: int = 5,
    ):
        self.confidence_threshold = confidence_threshold
        self.transition_frames = transition_frames
        self._transition_counter: Dict[int, int] = {}

    def blend(
        self,
        prev_output: np.ndarray,
        curr_output: np.ndarray,
        flow_magnitude: np.ndarray,
        track_id: int = 0,
    ) -> np.ndarray:
        """
        Blend prev and curr outputs guided by flow confidence.

        Args:
            prev_output:    Previous frame's output.
            curr_output:    Current frame's output.
            flow_magnitude: Per-pixel flow magnitude map.
            track_id:       Face tracking ID.

        Returns:
            Blended output frame.
        """
        # Normalise magnitude to [0, 1] confidence
        max_mag = flow_magnitude.max() if flow_magnitude.max() > 0 else 1.0
        confidence = 1.0 - np.clip(flow_magnitude / max_mag, 0, 1)

        # Average confidence
        avg_conf = float(confidence.mean())

        if avg_conf < self.confidence_threshold:
            # Large motion detected — start transition
            self._transition_counter[track_id] = self.transition_frames

        # During transition, ramp up blend toward current
        if self._transition_counter.get(track_id, 0) > 0:
            remaining = self._transition_counter[track_id]
            alpha = 1.0 - (remaining / self.transition_frames)
            self._transition_counter[track_id] -= 1
        else:
            alpha = 0.7  # Normally favor current

        result = cv2.addWeighted(
            curr_output.astype(np.float32),
            alpha,
            prev_output.astype(np.float32),
            1.0 - alpha,
            0,
        )
        return np.clip(result, 0, 255).astype(np.uint8)
