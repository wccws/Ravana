"""
High-level SDK API for face swapping.

As per PRD Section 9.1, this provides simple high-level functions:
- swap_image(source_img, target_img, config) -> output_img
- swap_video(source_img, input_video, config) -> output_video
- start_realtime_swap(source_img, camera_id, callback, config)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Union

import cv2
import numpy as np

from .core.types import Embedding, Frame
from .pipeline import FaceSwapPipeline, PipelineConfig


@dataclass
class FaceSwapConfig:
    """
    User-friendly configuration for face swapping.

    Simplified interface over PipelineConfig for ease of use.
    """

    # Quality/Speed tradeoff
    quality: str = "high"  # "low", "medium", "high"

    # Device
    device: str = "cuda"  # "cuda", "cpu"

    # Processing options
    color_correction: bool = True
    enable_smoothing: bool = True

    # Model paths (optional)
    detection_model_path: Optional[str] = None
    swap_model_path: Optional[str] = None

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to internal PipelineConfig."""
        config = PipelineConfig(
            device=self.device,
            color_correction=self.color_correction,
        )

        # Quality presets
        if self.quality == "low":
            config.crop_size = 128
            config.blend_mode = "alpha"
            config.enable_temporal = False
        elif self.quality == "medium":
            config.crop_size = 256
            config.blend_mode = "alpha"
            config.enable_temporal = self.enable_smoothing
        else:  # high
            config.crop_size = 256
            config.blend_mode = "feather"
            config.enable_temporal = self.enable_smoothing

        if self.swap_model_path:
            config.swap_model_path = self.swap_model_path

        return config


def swap_image(
    source_img: Union[str, Frame],
    target_img: Union[str, Frame],
    config: Optional[FaceSwapConfig] = None,
) -> Frame:
    """
    Swap a face from source image onto target image.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        target_img: Target image path or array to swap face onto
        config: Optional configuration

    Returns:
        Output image with swapped face

    Example:
        >>> result = swap_image("source.jpg", "target.jpg")
        >>> cv2.imwrite("output.jpg", result)
    """
    # Load images if paths provided
    if isinstance(source_img, str):
        source_img = cv2.imread(source_img)
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_img}")

    if isinstance(target_img, str):
        target_img = cv2.imread(target_img)
        if target_img is None:
            raise ValueError(f"Could not load target image: {target_img}")

    # Use default config if not provided
    cfg = config or FaceSwapConfig()

    # Create pipeline
    pipeline = FaceSwapPipeline(cfg.to_pipeline_config())

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Process target image
    result = pipeline.process_frame(target_img, source_embedding)

    return result


def swap_video(
    source_img: Union[str, Frame],
    input_video: str,
    output_video: str,
    config: Optional[FaceSwapConfig] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Swap a face from source image onto all frames of a video.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        input_video: Path to input video file
        output_video: Path to output video file
        config: Optional configuration
        progress_callback: Optional callback(frame_index, total_frames)

    Example:
        >>> swap_video("source.jpg", "input.mp4", "output.mp4")
    """
    # Load source image
    if isinstance(source_img, str):
        source_img = cv2.imread(source_img)
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_img}")

    # Use default config if not provided
    cfg = config or FaceSwapConfig()
    pipeline_config = cfg.to_pipeline_config()
    pipeline_config.enable_temporal = True  # Always enable for video

    # Create pipeline
    pipeline = FaceSwapPipeline(pipeline_config)

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Open video
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_video}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = pipeline.process_video_frame(frame, source_embedding, frame_idx)

            # Write output
            out.write(result)

            # Update progress
            if progress_callback:
                progress_callback(frame_idx, total_frames)

            frame_idx += 1

    finally:
        cap.release()
        out.release()


def start_realtime_swap(
    source_img: Union[str, Frame],
    camera_id: int = 0,
    callback: Optional[Callable[[Frame], None]] = None,
    config: Optional[FaceSwapConfig] = None,
    display_size: tuple = (1280, 720),
) -> None:
    """
    Start real-time face swapping from webcam.

    High-level API as per PRD Section 9.1.

    Args:
        source_img: Source image path or array containing the face to swap
        camera_id: Camera device ID (default 0 for primary webcam)
        callback: Optional callback for each processed frame
        config: Optional configuration
        display_size: Size to display the output

    Example:
        >>> start_realtime_swap("source.jpg", camera_id=0)
    """
    # Load source image
    if isinstance(source_img, str):
        source_img = cv2.imread(source_img)
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_img}")

    # Use fast config for real-time if not specified
    cfg = config or FaceSwapConfig(quality="medium", device="cuda")

    # Create pipeline
    pipeline_config = cfg.to_pipeline_config()
    pipeline_config.enable_temporal = True
    pipeline = FaceSwapPipeline(pipeline_config)

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Open camera
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Could not open camera: {camera_id}")

    # Set camera properties for performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, display_size[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, display_size[1])

    print("Starting real-time face swap. Press 'q' to quit.")

    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            result = pipeline.process_video_frame(frame, source_embedding, frame_idx)

            # Call callback if provided
            if callback:
                callback(result)

            # Display
            cv2.imshow("Face Swap - Press q to quit", result)

            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            frame_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()


def batch_swap(
    source_img: Union[str, Frame],
    target_paths: List[str],
    output_dir: str,
    config: Optional[FaceSwapConfig] = None,
) -> List[str]:
    """
    Batch process multiple images with the same source face.

    Args:
        source_img: Source image path or array
        target_paths: List of target image paths
        output_dir: Directory to save output images
        config: Optional configuration

    Returns:
        List of output file paths
    """
    from pathlib import Path

    # Load source image
    if isinstance(source_img, str):
        source_img = cv2.imread(source_img)
        if source_img is None:
            raise ValueError(f"Could not load source image: {source_img}")

    # Use default config
    cfg = config or FaceSwapConfig()
    pipeline = FaceSwapPipeline(cfg.to_pipeline_config())

    # Extract source embedding
    source_embedding = pipeline.extract_source_embedding(source_img)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_files = []

    for target_path in target_paths:
        # Load target
        target_img = cv2.imread(target_path)
        if target_img is None:
            print(f"Warning: Could not load {target_path}, skipping")
            continue

        # Process
        result = pipeline.process_frame(target_img, source_embedding)

        # Save
        output_file = output_path / f"swapped_{Path(target_path).name}"
        cv2.imwrite(str(output_file), result)
        output_files.append(str(output_file))

        print(f"Processed: {target_path} -> {output_file}")

    return output_files


from .core.types import AlignedFace, Embedding, FaceBBox, Landmarks, SwapResult

# Low-level API exports for advanced users
from .pipeline import FaceSwapPipeline, PipelineConfig

__all__ = [
    # High-level API
    "swap_image",
    "swap_video",
    "start_realtime_swap",
    "batch_swap",
    "FaceSwapConfig",
    # Low-level API
    "FaceSwapPipeline",
    "PipelineConfig",
    # Types
    "FaceBBox",
    "Landmarks",
    "AlignedFace",
    "Embedding",
    "SwapResult",
]
