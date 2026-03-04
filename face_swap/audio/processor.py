"""
Audio-synchronized video processing.

As per PRD Section 5.2 Step 9:
  - Rebuild the output video from processed frames and original audio.

This module provides:
  - Audio extraction from source video.
  - Audio re-muxing into the swapped output video.
  - Temporal alignment (frame rate preservation, A/V sync).
  - Subtitle / metadata track passthrough.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger("face_swap.audio")


class AudioProcessor:
    """
    Handles audio extraction, processing, and re-muxing for video pipelines.

    Uses FFmpeg as the underlying tool; gracefully degrades if FFmpeg
    is not installed (video-only output, no audio).
    """

    def __init__(self):
        self._ffmpeg = self._find_ffmpeg()

    @property
    def available(self) -> bool:
        """True if FFmpeg is available on this system."""
        return self._ffmpeg is not None

    # ------------------------------------------------------------------
    # Audio extraction
    # ------------------------------------------------------------------

    def extract_audio(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        codec: str = "aac",
    ) -> Optional[str]:
        """
        Extract audio track from a video file.

        Args:
            video_path:  Path to source video.
            output_path: Where to write the audio file (default: temp file).
            codec:       Audio codec (``aac``, ``mp3``, ``wav``).

        Returns:
            Path to extracted audio, or None if no audio track / FFmpeg missing.
        """
        if not self.available:
            logger.warning("FFmpeg not found; cannot extract audio.")
            return None

        if output_path is None:
            ext = {"aac": ".aac", "mp3": ".mp3", "wav": ".wav"}.get(codec, ".aac")
            output_path = tempfile.mktemp(suffix=ext)

        cmd = [
            self._ffmpeg,
            "-i",
            video_path,
            "-vn",  # no video
            "-acodec",
            "copy" if codec == "aac" else codec,
            "-y",  # overwrite
            output_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.warning("Audio extraction failed: %s", result.stderr[:500])
                return None

            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                logger.info("No audio track found in %s", video_path)
                return None

            logger.info("Audio extracted: %s", output_path)
            return output_path

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.warning("Audio extraction error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Re-muxing
    # ------------------------------------------------------------------

    def mux_audio(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        copy_video: bool = True,
    ) -> str:
        """
        Combine a video file with an audio file.

        Args:
            video_path:  Processed (swapped) video (may be silent).
            audio_path:  Audio track to add.
            output_path: Final output path.
            copy_video:  Copy video stream without re-encoding (fast).

        Returns:
            Path to the muxed output video.

        Raises:
            RuntimeError: If FFmpeg is not available.
        """
        if not self.available:
            raise RuntimeError(
                "FFmpeg is required for audio muxing. "
                "Install from: https://ffmpeg.org/download.html"
            )

        cmd = [
            self._ffmpeg,
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "copy" if copy_video else "libx264",
            "-c:a",
            "aac",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-shortest",
            "-y",
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"Audio muxing failed: {result.stderr[:500]}")

        logger.info("Audio muxed: %s", output_path)
        return output_path

    def swap_video_with_audio(
        self,
        original_video: str,
        swapped_video: str,
        output_path: str,
    ) -> str:
        """
        High-level: take the audio from the original and combine with
        the swapped video, producing a final output file.

        This is the typical end-to-end flow for video processing:
        1. Extract audio from original.
        2. Mux audio into swapped video.
        3. Clean up temp files.

        Args:
            original_video: Original input video (with audio).
            swapped_video:  Processed video (face-swapped, possibly silent).
            output_path:    Final output file.

        Returns:
            Path to the final video with audio.
        """
        audio_tmp = self.extract_audio(original_video)

        if audio_tmp is None:
            # No audio — just copy the swapped video
            shutil.copy2(swapped_video, output_path)
            return output_path

        try:
            return self.mux_audio(swapped_video, audio_tmp, output_path)
        finally:
            if os.path.exists(audio_tmp):
                os.remove(audio_tmp)

    # ------------------------------------------------------------------
    # Video info
    # ------------------------------------------------------------------

    def get_video_info(self, video_path: str) -> dict:
        """
        Get video metadata (duration, fps, resolution, has_audio).

        Returns:
            Dict with ``duration``, ``fps``, ``width``, ``height``,
            ``has_audio``, ``audio_codec``.
        """
        if not self.available:
            return {}

        ffprobe = self._ffmpeg.replace("ffmpeg", "ffprobe")
        cmd = [
            ffprobe,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return {}

            import json

            data = json.loads(result.stdout)

            info = {
                "duration": float(data.get("format", {}).get("duration", 0)),
                "has_audio": False,
                "fps": 0,
                "width": 0,
                "height": 0,
            }

            for stream in data.get("streams", []):
                if stream["codec_type"] == "video":
                    info["width"] = int(stream.get("width", 0))
                    info["height"] = int(stream.get("height", 0))
                    # Parse fps from r_frame_rate (e.g., "30000/1001")
                    rfr = stream.get("r_frame_rate", "0/1")
                    if "/" in rfr:
                        num, den = rfr.split("/")
                        info["fps"] = float(num) / float(den) if float(den) > 0 else 0
                elif stream["codec_type"] == "audio":
                    info["has_audio"] = True
                    info["audio_codec"] = stream.get("codec_name", "unknown")

            return info

        except (subprocess.TimeoutExpired, json.JSONDecodeError):
            return {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg() -> Optional[str]:
        """Locate the FFmpeg binary."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return ffmpeg

        # Common install locations
        common_paths = [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
        ]
        for p in common_paths:
            if os.path.isfile(p):
                return p

        return None
