"""
ONNX → TensorRT engine exporter.

As per PRD Section 5.9, this converts ONNX models into TensorRT engines
for deployment builds, supporting FP32, FP16, and INT8 precision modes.

TensorRT engines are hardware-specific; the export must be run on the
same GPU architecture that will serve inference.

Usage:
    >>> from face_swap.optimization import TensorRTExporter, ExportConfig
    >>> exporter = TensorRTExporter()
    >>> exporter.export(
    ...     onnx_path="models/inswapper_128.onnx",
    ...     engine_path="models/inswapper_128_fp16.engine",
    ...     config=ExportConfig(precision="fp16"),
    ... )
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("face_swap.optimization.export")


@dataclass
class ExportConfig:
    """Configuration for TensorRT export.

    Attributes:
        precision:        Inference precision — ``fp32``, ``fp16``, or ``int8``.
        max_workspace_gb: Maximum GPU workspace for TensortRT builder (GB).
        max_batch_size:   Maximum batch size the engine should support.
        dynamic_axes:     Whether to build with dynamic batch dimension.
        calibration_data: Path to calibration images folder (required for INT8).
        calibration_count: Number of images to use for INT8 calibration.
        verbose:          Enable builder verbose logging.
    """

    precision: str = "fp16"
    max_workspace_gb: float = 4.0
    max_batch_size: int = 1
    dynamic_axes: bool = False
    calibration_data: Optional[str] = None
    calibration_count: int = 500
    verbose: bool = False


class TensorRTExporter:
    """
    Exports ONNX models to TensorRT serialised engines.

    Supports:
      - FP32, FP16, and INT8 precision.
      - Dynamic batch sizes via optimisation profiles.
      - INT8 calibration using representative image data.
    """

    def __init__(self):
        self._trt = None
        self._ensure_tensorrt()

    def _ensure_tensorrt(self):
        """Lazy-import TensorRT (so the rest of the SDK works without it)."""
        try:
            import tensorrt as trt

            self._trt = trt
            self._logger = trt.Logger(trt.Logger.INFO)
        except ImportError:
            self._trt = None
            self._logger = None

    @property
    def available(self) -> bool:
        """True if TensorRT is installed and importable."""
        return self._trt is not None

    def export(
        self,
        onnx_path: str,
        engine_path: str,
        config: Optional[ExportConfig] = None,
    ) -> str:
        """
        Export an ONNX model to a TensortRT engine.

        Args:
            onnx_path:   Path to the source ``.onnx`` model.
            engine_path: Destination path for the serialised ``.engine`` file.
            config:      Export configuration.

        Returns:
            Path to the written engine file.

        Raises:
            ImportError:  If TensorRT is not installed.
            RuntimeError: If the build fails.
            FileNotFoundError: If the ONNX file does not exist.
        """
        if not self.available:
            raise ImportError(
                "TensorRT is required for engine export. "
                "Install with: pip install tensorrt"
            )
        trt = self._trt

        cfg = config or ExportConfig()

        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # Update logger verbosity
        if cfg.verbose:
            self._logger = trt.Logger(trt.Logger.VERBOSE)

        logger.info(
            "Exporting %s → %s (precision=%s)", onnx_path, engine_path, cfg.precision
        )
        start = time.time()

        # ── Parse ONNX ─────────────────────────────────────────────────
        EXPLICIT_BATCH = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        builder = trt.Builder(self._logger)
        network = builder.create_network(EXPLICIT_BATCH)
        parser = trt.OnnxParser(network, self._logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError(f"Failed to parse ONNX model: {onnx_path}")

        # Print network info
        n_inputs = network.num_inputs
        n_outputs = network.num_outputs
        logger.info(
            "Network: %d inputs, %d outputs, %d layers",
            n_inputs,
            n_outputs,
            network.num_layers,
        )
        for i in range(n_inputs):
            inp = network.get_input(i)
            logger.info("  Input  %d: %-20s  shape=%s", i, inp.name, inp.shape)
        for i in range(n_outputs):
            out = network.get_output(i)
            logger.info("  Output %d: %-20s  shape=%s", i, out.name, out.shape)

        # ── Builder config ─────────────────────────────────────────────
        builder_cfg = builder.create_builder_config()
        workspace_bytes = int(cfg.max_workspace_gb * (1 << 30))
        builder_cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

        # Precision
        if cfg.precision == "fp16":
            if not builder.platform_has_fast_fp16:
                logger.warning("GPU does not have fast FP16 — falling back to FP32.")
            else:
                builder_cfg.set_flag(trt.BuilderFlag.FP16)
                logger.info("FP16 enabled.")

        elif cfg.precision == "int8":
            if not builder.platform_has_fast_int8:
                raise RuntimeError("GPU does not support fast INT8.")
            builder_cfg.set_flag(trt.BuilderFlag.INT8)
            # Also enable FP16 as fallback for layers that can't run INT8
            if builder.platform_has_fast_fp16:
                builder_cfg.set_flag(trt.BuilderFlag.FP16)

            # INT8 calibrator
            if cfg.calibration_data is None:
                raise ValueError(
                    "INT8 precision requires calibration_data path " "in ExportConfig."
                )
            calibrator = _ImageCalibrator(
                data_dir=cfg.calibration_data,
                max_count=cfg.calibration_count,
                input_shape=tuple(network.get_input(0).shape),
            )
            builder_cfg.int8_calibrator = calibrator
            logger.info("INT8 enabled with calibration from %s", cfg.calibration_data)

        # Dynamic batch via optimisation profile
        if cfg.dynamic_axes:
            profile = builder.create_optimization_profile()
            for i in range(n_inputs):
                inp = network.get_input(i)
                shape = list(inp.shape)
                # Replace batch dimension with min/opt/max
                min_shape = [1] + shape[1:]
                opt_shape = [cfg.max_batch_size] + shape[1:]
                max_shape = [cfg.max_batch_size] + shape[1:]
                profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
            builder_cfg.add_optimization_profile(profile)

        # ── Build engine ───────────────────────────────────────────────
        serialized = builder.build_serialized_network(network, builder_cfg)
        if serialized is None:
            raise RuntimeError("TensorRT build failed — no serialised engine produced.")

        # ── Write to disk ──────────────────────────────────────────────
        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized)

        elapsed = time.time() - start
        size_mb = os.path.getsize(engine_path) / (1 << 20)
        logger.info(
            "Engine written: %s (%.1f MB) in %.1f s", engine_path, size_mb, elapsed
        )

        return engine_path


# ── INT8 calibrator ─────────────────────────────────────────────────────


class _ImageCalibrator:
    """
    Simple INT8 calibrator that reads images from a directory.

    TensorRT calls ``get_batch`` repeatedly during calibration.
    """

    def __init__(
        self,
        data_dir: str,
        max_count: int = 500,
        input_shape: tuple = (1, 3, 128, 128),
    ):
        import glob

        self.batch_size = input_shape[0] if input_shape[0] > 0 else 1
        self.input_shape = input_shape
        self._images = sorted(
            glob.glob(os.path.join(data_dir, "*.jpg"))
            + glob.glob(os.path.join(data_dir, "*.png"))
        )[:max_count]
        self._idx = 0
        self._device_input = None

    def get_batch_size(self) -> int:
        return self.batch_size

    def get_batch(self, names: list) -> Optional[list]:
        if self._idx >= len(self._images):
            return None

        import cv2

        try:
            import pycuda.driver as cuda
        except ImportError:
            logger.warning("pycuda required for INT8 calibration.")
            return None

        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.imread(self._images[self._idx])
        img = cv2.resize(img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)

        if self._device_input is None:
            self._device_input = cuda.mem_alloc(img.nbytes)

        cuda.memcpy_htod(self._device_input, img)
        self._idx += 1
        return [int(self._device_input)]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        pass
