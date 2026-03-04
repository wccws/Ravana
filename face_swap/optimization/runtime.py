"""
TensorRT inference runtime.

As per PRD Section 5.9, this provides a TensorRT-accelerated inference
engine that can be used as a drop-in replacement for ONNX Runtime in
deployment builds for maximum real-time performance.

The runtime handles:
  - Loading serialised TensorRT engines (.engine files).
  - GPU memory allocation and CUDA stream management.
  - Synchronous and asynchronous inference.
  - Graceful fallback to ONNX Runtime when TensorRT is unavailable.

Usage:
    >>> from face_swap.optimization import TensorRTRuntime
    >>> rt = TensorRTRuntime("models/inswapper_128_fp16.engine")
    >>> output = rt.infer(target_tensor, embedding_tensor)
"""

import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger("face_swap.optimization.runtime")


class TensorRTRuntime:
    """
    TensorRT engine loader and inference runtime.

    Manages the full lifecycle of a TensorRT inference session:
    engine deserialisation → context creation → buffer alloc → inference.
    """

    def __init__(
        self,
        engine_path: str,
        device_id: int = 0,
        max_batch_size: int = 1,
    ):
        """
        Load a serialised TensorRT engine.

        Args:
            engine_path:    Path to the ``.engine`` file.
            device_id:      CUDA device index.
            max_batch_size: Maximum batch size for buffer allocation.

        Raises:
            ImportError:       If TensorRT or pycuda is not installed.
            FileNotFoundError: If the engine file does not exist.
        """
        self._engine_path = engine_path
        self._device_id = device_id
        self._max_batch = max_batch_size

        # Lazy state (initialised on first infer or explicit .load())
        self._trt = None
        self._engine = None
        self._context = None
        self._stream = None
        self._buffers: Dict[str, dict] = {}  # name → {host, device, shape}
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Deserialise engine and allocate buffers."""
        if self._loaded:
            return

        try:
            import pycuda.autoinit  # noqa: F401 — initialises CUDA context
            import pycuda.driver as cuda
            import tensorrt as trt
        except ImportError as e:
            raise ImportError(
                "TensorRT and pycuda are required. "
                "Install with: pip install tensorrt pycuda"
            ) from e

        self._trt = trt
        self._cuda = cuda

        if not os.path.exists(self._engine_path):
            raise FileNotFoundError(f"Engine not found: {self._engine_path}")

        # Deserialise
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(self._engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        if self._engine is None:
            raise RuntimeError(f"Failed to load engine: {self._engine_path}")

        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Allocate host + device buffers for every binding
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            shape = list(self._engine.get_binding_shape(i))

            # Replace -1 (dynamic) with max_batch
            shape = [self._max_batch if s == -1 else s for s in shape]

            size = int(np.prod(shape))
            host_buf = np.empty(size, dtype=dtype)
            dev_buf = cuda.mem_alloc(host_buf.nbytes)

            is_input = self._engine.binding_is_input(i)
            self._buffers[name] = {
                "host": host_buf,
                "device": dev_buf,
                "shape": tuple(shape),
                "dtype": dtype,
                "is_input": is_input,
                "index": i,
            }

        self._loaded = True
        logger.info(
            "Loaded TensorRT engine: %s (%d bindings)",
            self._engine_path,
            self._engine.num_bindings,
        )

    def unload(self) -> None:
        """Free GPU memory and release the engine."""
        self._buffers.clear()
        self._context = None
        self._engine = None
        self._stream = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run synchronous inference.

        Args:
            **inputs: Named input arrays matching the engine bindings.
                      e.g. ``infer(target=target_tensor, source=embedding_tensor)``

        Returns:
            Dict mapping output binding names to numpy result arrays.

        Example:
            >>> outputs = rt.infer(target=target_img, source=id_vec)
            >>> swapped = outputs["output"]
        """
        if not self._loaded:
            self.load()

        cuda = self._cuda

        # Copy inputs from host to device
        for name, arr in inputs.items():
            if name not in self._buffers:
                raise ValueError(f"Unknown input binding: {name}")
            buf = self._buffers[name]
            np.copyto(buf["host"][: arr.size], arr.ravel())
            cuda.memcpy_htod_async(buf["device"], buf["host"], self._stream)

        # Build bindings list (ordered by index)
        bindings = [None] * self._engine.num_bindings
        for buf in self._buffers.values():
            bindings[buf["index"]] = int(buf["device"])

        # Execute
        self._context.execute_async_v2(
            bindings=bindings, stream_handle=self._stream.handle
        )

        # Copy outputs from device to host
        outputs: Dict[str, np.ndarray] = {}
        for name, buf in self._buffers.items():
            if not buf["is_input"]:
                cuda.memcpy_dtoh_async(buf["host"], buf["device"], self._stream)

        self._stream.synchronize()

        for name, buf in self._buffers.items():
            if not buf["is_input"]:
                outputs[name] = buf["host"].reshape(buf["shape"]).copy()

        return outputs

    def infer_numpy(
        self,
        *args: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Positional-argument variant of :meth:`infer`.

        Inputs are matched to bindings in definition order.

        Returns:
            List of output arrays in binding order.
        """
        if not self._loaded:
            self.load()

        # Map args to input binding names (in order)
        input_names = [
            name
            for name, buf in sorted(self._buffers.items(), key=lambda x: x[1]["index"])
            if buf["is_input"]
        ]

        if len(args) != len(input_names):
            raise ValueError(f"Expected {len(input_names)} inputs, got {len(args)}")

        kw = dict(zip(input_names, args))
        result = self.infer(**kw)

        # Return outputs in binding order
        output_names = [
            name
            for name, buf in sorted(self._buffers.items(), key=lambda x: x[1]["index"])
            if not buf["is_input"]
        ]
        return [result[n] for n in output_names]

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def binding_info(self) -> Dict[str, dict]:
        """Return metadata for every engine binding."""
        if not self._loaded:
            self.load()
        return {
            name: {
                "shape": b["shape"],
                "dtype": str(b["dtype"]),
                "is_input": b["is_input"],
            }
            for name, b in self._buffers.items()
        }

    def benchmark(self, warmup: int = 10, iterations: int = 100) -> dict:
        """
        Run a latency micro-benchmark.

        Args:
            warmup:     Number of warm-up iterations.
            iterations: Number of timed iterations.

        Returns:
            Dict with ``avg_ms``, ``min_ms``, ``max_ms``, ``fps``.
        """
        if not self._loaded:
            self.load()

        # Build dummy inputs
        dummy_inputs = {}
        for name, buf in self._buffers.items():
            if buf["is_input"]:
                dummy_inputs[name] = np.random.randn(*buf["shape"]).astype(buf["dtype"])

        # Warm up
        for _ in range(warmup):
            self.infer(**dummy_inputs)

        # Timed runs
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.infer(**dummy_inputs)
            times.append((time.perf_counter() - t0) * 1000)

        avg = sum(times) / len(times)
        return {
            "avg_ms": avg,
            "min_ms": min(times),
            "max_ms": max(times),
            "fps": 1000.0 / avg if avg > 0 else 0,
            "iterations": iterations,
        }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_exc):
        self.unload()

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"<TensorRTRuntime engine={self._engine_path!r} ({status})>"


class OnnxFallbackRuntime:
    """
    Drop-in replacement that uses ONNX Runtime when TensorRT is unavailable.

    API is identical to ``TensorRTRuntime`` so callers can swap seamlessly.
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        self._model_path = model_path
        self._device = device
        self._session = None

    def load(self) -> None:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider"]
            if self._device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self._session = ort.InferenceSession(self._model_path, providers=providers)

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def infer(self, **inputs: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.loaded:
            self.load()
        result = self._session.run(None, inputs)
        output_names = [o.name for o in self._session.get_outputs()]
        return dict(zip(output_names, result))

    def infer_numpy(self, *args: np.ndarray) -> List[np.ndarray]:
        if not self.loaded:
            self.load()
        input_names = [i.name for i in self._session.get_inputs()]
        kw = dict(zip(input_names, args))
        return list(self._session.run(None, kw))

    def unload(self) -> None:
        self._session = None

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_exc):
        self.unload()


def get_best_runtime(
    onnx_path: str,
    engine_path: Optional[str] = None,
    device: str = "cuda",
):
    """
    Return the best available runtime.

    Prefers TensorRT if an engine file exists and TensorRT is installed;
    otherwise falls back to ONNX Runtime.

    Args:
        onnx_path:   Path to the ONNX model.
        engine_path: Path to a pre-built TensorRT engine (may be None).
        device:      ``cuda`` or ``cpu``.

    Returns:
        A ``TensorRTRuntime`` or ``OnnxFallbackRuntime`` instance.
    """
    if engine_path and os.path.exists(engine_path):
        try:
            return TensorRTRuntime(engine_path)
        except ImportError:
            logger.info("TensorRT not available; falling back to ONNX Runtime.")

    return OnnxFallbackRuntime(onnx_path, device=device)
