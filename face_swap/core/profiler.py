"""
Performance profiler and benchmarking utilities.

As per PRD Section 2.2 success metrics, this provides tools to measure:
  - Per-frame end-to-end latency.
  - Per-stage breakdown (detection, landmarks, alignment, swap, blend).
  - FPS tracking over time windows.
  - Throughput (offline batch).

Integrators can use this to validate that the system meets the
≤ 40 ms per-frame target on their hardware.
"""

import json
import logging
import statistics
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger("face_swap.profiler")


@dataclass
class StageTimings:
    """Timing breakdown for a single frame."""

    detection_ms: float = 0.0
    landmarks_ms: float = 0.0
    alignment_ms: float = 0.0
    embedding_ms: float = 0.0
    swap_ms: float = 0.0
    blend_ms: float = 0.0
    temporal_ms: float = 0.0
    watermark_ms: float = 0.0
    total_ms: float = 0.0
    num_faces: int = 0

    @property
    def meets_realtime_target(self) -> bool:
        """True if total latency is within the 40 ms PRD target."""
        return self.total_ms <= 40.0

    def to_dict(self) -> dict:
        return self.__dict__


@dataclass
class BenchmarkReport:
    """Aggregate statistics over a benchmark run."""

    num_frames: int = 0
    avg_total_ms: float = 0.0
    p50_total_ms: float = 0.0
    p95_total_ms: float = 0.0
    p99_total_ms: float = 0.0
    avg_fps: float = 0.0
    avg_stage_ms: Dict[str, float] = field(default_factory=dict)
    meets_target_pct: float = 0.0  # % of frames ≤ 40 ms

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.__dict__, indent=indent)


class PipelineProfiler:
    """
    Lightweight profiler that wraps around pipeline stages.

    Usage:
        >>> profiler = PipelineProfiler()
        >>> with profiler.stage("detection"):
        ...     bboxes = detector.detect(frame)
        >>> profiler.end_frame()
        >>> report = profiler.report()
    """

    def __init__(self, window_size: int = 300):
        """
        Args:
            window_size: Number of frames to keep for rolling statistics.
        """
        self._window_size = window_size
        self._history: deque = deque(maxlen=window_size)
        self._current: Dict[str, float] = {}
        self._frame_start: Optional[float] = None
        self._active_stage: Optional[str] = None
        self._stage_start: Optional[float] = None
        self._num_faces_current = 0
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    # ------------------------------------------------------------------
    # Per-frame API
    # ------------------------------------------------------------------

    def begin_frame(self) -> None:
        """Call at the very start of frame processing."""
        if not self._enabled:
            return
        self._frame_start = time.perf_counter()
        self._current = {}
        self._num_faces_current = 0

    @contextmanager
    def stage(self, name: str):
        """Context manager to time a pipeline stage.

        Example:
            with profiler.stage("detection"):
                ...
        """
        if not self._enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._current[name] = self._current.get(name, 0) + elapsed

    def set_num_faces(self, n: int) -> None:
        self._num_faces_current = n

    def end_frame(self) -> StageTimings:
        """Call after all stages are done for the current frame.

        Returns a ``StageTimings`` instance for the just-completed frame.
        """
        if not self._enabled:
            return StageTimings()

        total = (
            time.perf_counter() - (self._frame_start or time.perf_counter())
        ) * 1000

        timings = StageTimings(
            detection_ms=self._current.get("detection", 0),
            landmarks_ms=self._current.get("landmarks", 0),
            alignment_ms=self._current.get("alignment", 0),
            embedding_ms=self._current.get("embedding", 0),
            swap_ms=self._current.get("swap", 0),
            blend_ms=self._current.get("blend", 0),
            temporal_ms=self._current.get("temporal", 0),
            watermark_ms=self._current.get("watermark", 0),
            total_ms=total,
            num_faces=self._num_faces_current,
        )

        self._history.append(timings)
        return timings

    # ------------------------------------------------------------------
    # Aggregate reporting
    # ------------------------------------------------------------------

    def report(self) -> BenchmarkReport:
        """Generate aggregate statistics over the recorded history."""
        n = len(self._history)
        if n == 0:
            return BenchmarkReport()

        totals = [t.total_ms for t in self._history]
        totals_sorted = sorted(totals)

        stage_names = [
            "detection",
            "landmarks",
            "alignment",
            "embedding",
            "swap",
            "blend",
            "temporal",
            "watermark",
        ]
        avg_stages = {}
        for s in stage_names:
            vals = [getattr(t, f"{s}_ms") for t in self._history]
            avg_stages[s] = statistics.mean(vals)

        meets = sum(1 for t in totals if t <= 40.0)

        return BenchmarkReport(
            num_frames=n,
            avg_total_ms=statistics.mean(totals),
            p50_total_ms=totals_sorted[n // 2],
            p95_total_ms=totals_sorted[int(n * 0.95)],
            p99_total_ms=totals_sorted[int(n * 0.99)],
            avg_fps=(
                1000.0 / statistics.mean(totals) if statistics.mean(totals) > 0 else 0
            ),
            avg_stage_ms=avg_stages,
            meets_target_pct=100.0 * meets / n,
        )

    def latest(self) -> Optional[StageTimings]:
        """Return the most recent frame timings."""
        return self._history[-1] if self._history else None

    def avg_fps(self) -> float:
        """Rolling average FPS."""
        if not self._history:
            return 0.0
        avg_ms = statistics.mean(t.total_ms for t in self._history)
        return 1000.0 / avg_ms if avg_ms > 0 else 0.0

    def reset(self) -> None:
        """Clear all recorded history."""
        self._history.clear()
        self._current = {}
