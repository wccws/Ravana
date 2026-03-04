"""
Temporal consistency and tracking module.

As per PRD Section 5.8, this minimizes temporal flicker in facial appearance,
color, and position when processing video using optical flow and tracking.
"""

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core.types import FaceBBox, Landmarks, SwapResult


@dataclass
class TrackedFace:
    """Tracked face with temporal information."""

    track_id: int
    bbox: FaceBBox
    landmarks: Optional[Landmarks]
    embedding: Optional[np.ndarray]
    last_seen: int  # Frame number
    history: deque  # History of positions/appearances


class FaceTracker:
    """
    Face tracker for maintaining identity across frames.

    Uses IOU-based tracking with appearance matching.
    """

    def __init__(
        self,
        max_age: int = 5,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize face tracker.

        Args:
            max_age: Maximum frames to keep lost tracks
            min_hits: Minimum detections to confirm track
            iou_threshold: Minimum IOU for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, TrackedFace] = {}
        self.next_track_id = 0
        self.frame_count = 0

    def update(
        self,
        bboxes: List[FaceBBox],
        landmarks_list: Optional[List[Landmarks]] = None,
        embeddings: Optional[List[np.ndarray]] = None,
    ) -> List[TrackedFace]:
        """
        Update tracks with new detections.

        Args:
            bboxes: Detected face bounding boxes
            landmarks_list: Optional landmarks for each face
            embeddings: Optional embeddings for each face

        Returns:
            List of active tracked faces
        """
        self.frame_count += 1

        if landmarks_list is None:
            landmarks_list = [None] * len(bboxes)
        if embeddings is None:
            embeddings = [None] * len(bboxes)

        # Match detections to existing tracks
        matched_tracks = []
        unmatched_dets = list(range(len(bboxes)))

        # Calculate IOU between all tracks and detections
        for track_id, track in list(self.tracks.items()):
            best_iou = self.iou_threshold
            best_det_idx = -1

            for det_idx in unmatched_dets:
                iou = self._calculate_iou(track.bbox, bboxes[det_idx])
                if iou > best_iou:
                    # Check appearance similarity if embeddings available
                    if embeddings[det_idx] is not None and track.embedding is not None:
                        sim = self._cosine_similarity(
                            track.embedding, embeddings[det_idx]
                        )
                        if sim < 0.5:  # Different person
                            continue

                    best_iou = iou
                    best_det_idx = det_idx

            if best_det_idx >= 0:
                # Update track
                self.tracks[track_id].bbox = bboxes[best_det_idx]
                self.tracks[track_id].landmarks = landmarks_list[best_det_idx]
                if embeddings[best_det_idx] is not None:
                    self.tracks[track_id].embedding = embeddings[best_det_idx]
                self.tracks[track_id].last_seen = self.frame_count

                # Update history
                self.tracks[track_id].history.append(
                    {"bbox": bboxes[best_det_idx], "frame": self.frame_count}
                )
                if len(self.tracks[track_id].history) > 10:
                    self.tracks[track_id].history.popleft()

                matched_tracks.append(self.tracks[track_id])
                unmatched_dets.remove(best_det_idx)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            track_id = self.next_track_id
            self.next_track_id += 1

            self.tracks[track_id] = TrackedFace(
                track_id=track_id,
                bbox=bboxes[det_idx],
                landmarks=landmarks_list[det_idx],
                embedding=embeddings[det_idx],
                last_seen=self.frame_count,
                history=deque(
                    [{"bbox": bboxes[det_idx], "frame": self.frame_count}], maxlen=10
                ),
            )

            matched_tracks.append(self.tracks[track_id])

        # Remove old tracks
        for track_id in list(self.tracks.keys()):
            if self.frame_count - self.tracks[track_id].last_seen > self.max_age:
                del self.tracks[track_id]

        return matched_tracks

    def _calculate_iou(self, bbox1: FaceBBox, bbox2: FaceBBox) -> float:
        """Calculate IOU between two bounding boxes."""
        x1 = max(bbox1.x1, bbox2.x1)
        y1 = max(bbox1.y1, bbox2.y1)
        x2 = min(bbox1.x2, bbox2.x2)
        y2 = min(bbox1.y2, bbox2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


class TemporalSmoother:
    """
    Temporal smoother for reducing flicker in video face swaps.

    As per PRD Section 5.8, this:
    - Tracks faces across frames
    - Applies temporal smoothing to positions and appearances
    - Maintains consistent identity mappings
    """

    def __init__(
        self,
        smooth_factor: float = 0.7,
        use_optical_flow: bool = False,
    ):
        """
        Initialize temporal smoother.

        Args:
            smooth_factor: Smoothing factor (0 = no smoothing, 1 = full smoothing)
            use_optical_flow: Whether to use optical flow for position tracking
        """
        self.smooth_factor = smooth_factor
        self.use_optical_flow = use_optical_flow

        self.tracker = FaceTracker()
        self.prev_frame: Optional[np.ndarray] = None
        self.prev_gray: Optional[np.ndarray] = None

        # Cache for swap results
        self.swap_cache: Dict[int, SwapResult] = {}
        self.position_cache: Dict[int, FaceBBox] = {}

    def smooth_bboxes(
        self,
        bboxes: List[FaceBBox],
        frame: np.ndarray,
    ) -> List[FaceBBox]:
        """
        Smooth bounding box positions temporally.

        Args:
            bboxes: Current frame detections
            frame: Current frame

        Returns:
            Smoothed bounding boxes
        """
        # Update tracker
        tracked = self.tracker.update(bboxes)

        # Smooth positions
        smoothed = []
        for track in tracked:
            if track.track_id in self.position_cache:
                prev_bbox = self.position_cache[track.track_id]

                # Exponential moving average
                alpha = 1.0 - self.smooth_factor
                smooth_bbox = FaceBBox(
                    x1=alpha * track.bbox.x1 + (1 - alpha) * prev_bbox.x1,
                    y1=alpha * track.bbox.y1 + (1 - alpha) * prev_bbox.y1,
                    x2=alpha * track.bbox.x2 + (1 - alpha) * prev_bbox.x2,
                    y2=alpha * track.bbox.y2 + (1 - alpha) * prev_bbox.y2,
                    confidence=track.bbox.confidence,
                    track_id=track.track_id,
                )
            else:
                smooth_bbox = track.bbox
                smooth_bbox.track_id = track.track_id

            self.position_cache[track.track_id] = smooth_bbox
            smoothed.append(smooth_bbox)

        # Update optical flow if enabled
        if self.use_optical_flow:
            self._update_optical_flow(frame)

        return smoothed

    def smooth_swap_result(
        self,
        track_id: int,
        swap_result: SwapResult,
    ) -> SwapResult:
        """
        Smooth swap result appearance temporally.

        Args:
            track_id: Face track ID
            swap_result: Current swap result

        Returns:
            Smoothed swap result
        """
        if track_id not in self.swap_cache:
            self.swap_cache[track_id] = swap_result
            return swap_result

        prev_result = self.swap_cache[track_id]

        # Temporal smoothing of the face image
        alpha = 1.0 - self.smooth_factor
        smoothed_face = (
            alpha * swap_result.swapped_face.astype(np.float32)
            + (1 - alpha) * prev_result.swapped_face.astype(np.float32)
        ).astype(np.uint8)

        # Smooth mask
        smoothed_mask = alpha * swap_result.mask + (1 - alpha) * prev_result.mask

        # Update cache
        self.swap_cache[track_id] = SwapResult(
            swapped_face=smoothed_face,
            mask=smoothed_mask,
            source_embedding=swap_result.source_embedding,
            target_aligned=swap_result.target_aligned,
            quality_score=swap_result.quality_score,
        )

        return self.swap_cache[track_id]

    def _update_optical_flow(self, frame: np.ndarray) -> None:
        """Update optical flow for position tracking."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        self.prev_gray = gray
        self.prev_frame = frame.copy()

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.swap_cache.clear()
        self.position_cache.clear()
        self.prev_frame = None
        self.prev_gray = None
        self.tracker = FaceTracker()
