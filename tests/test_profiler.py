"""
Tests for the profiler module.
"""

import time

import pytest

from face_swap.core.profiler import BenchmarkReport, PipelineProfiler, StageTimings


@pytest.fixture
def profiler():
    return PipelineProfiler(window_size=100)


class TestStageTimings:
    def test_meets_target(self):
        t = StageTimings(total_ms=30.0)
        assert t.meets_realtime_target is True

    def test_exceeds_target(self):
        t = StageTimings(total_ms=50.0)
        assert t.meets_realtime_target is False


class TestPipelineProfiler:
    def test_basic_flow(self, profiler):
        profiler.begin_frame()
        with profiler.stage("detection"):
            time.sleep(0.001)
        with profiler.stage("swap"):
            time.sleep(0.001)
        timings = profiler.end_frame()
        assert timings.total_ms > 0
        assert timings.detection_ms > 0

    def test_report(self, profiler):
        for _ in range(10):
            profiler.begin_frame()
            with profiler.stage("detection"):
                pass
            profiler.end_frame()
        report = profiler.report()
        assert report.num_frames == 10
        assert report.avg_total_ms >= 0

    def test_disabled(self, profiler):
        profiler.enabled = False
        profiler.begin_frame()
        with profiler.stage("detection"):
            pass
        timings = profiler.end_frame()
        assert timings.total_ms == 0

    def test_avg_fps(self, profiler):
        for _ in range(5):
            profiler.begin_frame()
            time.sleep(0.01)
            profiler.end_frame()
        fps = profiler.avg_fps()
        assert fps > 0

    def test_reset(self, profiler):
        profiler.begin_frame()
        profiler.end_frame()
        profiler.reset()
        assert profiler.latest() is None


class TestBenchmarkReport:
    def test_json(self):
        r = BenchmarkReport(num_frames=100, avg_total_ms=25.5, avg_fps=39.2)
        js = r.to_json()
        assert "100" in js
        assert "25.5" in js
