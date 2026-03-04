"""
Python ctypes wrapper for the native C API.

This allows the same compiled shared library (face_swap.dll / libface_swap.so)
to be consumed from Python, which is useful for:
  - Testing the C API without a C compiler.
  - Hybrid deployments where some parts run natively.
  - Benchmarking native vs. pure-Python performance.

Usage:
    >>> from face_swap.native import NativeFaceSwap
    >>> fs = NativeFaceSwap("./build/face_swap.dll")
    >>> fs.init()
    >>> session = fs.create_session()
    >>> fs.set_source_file(session, "source.jpg")
    >>> fs.swap_image_file(session, "target.jpg", "output.jpg")
    >>> fs.destroy_session(session)
    >>> fs.shutdown()
"""

import ctypes
import ctypes.util
import platform
from pathlib import Path
from typing import Optional

import numpy as np

# ── C type mirrors ───────────────────────────────────────────────────────


class _FsConfig(ctypes.Structure):
    _fields_ = [
        ("device", ctypes.c_int),
        ("quality", ctypes.c_int),
        ("blend_mode", ctypes.c_int),
        ("color_correction", ctypes.c_int),
        ("enable_temporal", ctypes.c_int),
        ("async_detection", ctypes.c_int),
        ("enable_watermark", ctypes.c_int),
        ("crop_size", ctypes.c_int),
        ("swap_model_path", ctypes.c_char_p),
        ("detection_model_path", ctypes.c_char_p),
    ]


class _FsBBox(ctypes.Structure):
    _fields_ = [
        ("x1", ctypes.c_float),
        ("y1", ctypes.c_float),
        ("x2", ctypes.c_float),
        ("y2", ctypes.c_float),
        ("confidence", ctypes.c_float),
        ("track_id", ctypes.c_int),
    ]


class _FsImage(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("pixel_format", ctypes.c_int),
    ]


class _FsImageMut(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("stride", ctypes.c_int),
        ("pixel_format", ctypes.c_int),
    ]


class _FsTimings(ctypes.Structure):
    _fields_ = [
        ("detection_ms", ctypes.c_float),
        ("landmarks_ms", ctypes.c_float),
        ("alignment_ms", ctypes.c_float),
        ("swap_ms", ctypes.c_float),
        ("blend_ms", ctypes.c_float),
        ("total_ms", ctypes.c_float),
        ("num_faces", ctypes.c_int),
    ]


# ── Constants ────────────────────────────────────────────────────────────

FS_OK = 0
FS_DEVICE_CPU = 0
FS_DEVICE_CUDA = 1
FS_QUALITY_LOW = 0
FS_QUALITY_MEDIUM = 1
FS_QUALITY_HIGH = 2
FS_BLEND_ALPHA = 0
FS_BLEND_POISSON = 1
FS_BLEND_FEATHER = 2
FS_PIXEL_BGR = 0
FS_PIXEL_RGB = 1


# ── Wrapper class ────────────────────────────────────────────────────────


class NativeFaceSwap:
    """High-level Python wrapper around the native C API."""

    def __init__(self, lib_path: Optional[str] = None):
        """
        Load the native shared library.

        Args:
            lib_path: Explicit path to face_swap.dll / libface_swap.so.
                      If None, will attempt to find it automatically.
        """
        if lib_path is None:
            lib_path = self._find_library()

        self._lib = ctypes.CDLL(lib_path)
        self._setup_signatures()

    # ── Lifecycle ────────────────────────────────────────────────────────

    def init(self) -> None:
        self._check(self._lib.fs_init())

    def shutdown(self) -> None:
        self._check(self._lib.fs_shutdown())

    def version(self) -> str:
        return self._lib.fs_version().decode("utf-8")

    # ── Session ──────────────────────────────────────────────────────────

    def create_session(
        self,
        *,
        device: int = FS_DEVICE_CUDA,
        quality: int = FS_QUALITY_HIGH,
        blend_mode: int = FS_BLEND_ALPHA,
        crop_size: int = 256,
        color_correction: bool = True,
        enable_temporal: bool = True,
        swap_model_path: Optional[str] = None,
    ) -> ctypes.c_void_p:
        """Create a session and return its opaque handle."""
        cfg = _FsConfig()
        cfg.device = device
        cfg.quality = quality
        cfg.blend_mode = blend_mode
        cfg.crop_size = crop_size
        cfg.color_correction = int(color_correction)
        cfg.enable_temporal = int(enable_temporal)
        cfg.async_detection = 0
        cfg.enable_watermark = 0
        cfg.swap_model_path = (
            swap_model_path.encode("utf-8") if swap_model_path else None
        )
        cfg.detection_model_path = None

        handle = ctypes.c_void_p()
        self._check(
            self._lib.fs_session_create(ctypes.byref(handle), ctypes.byref(cfg))
        )
        return handle

    def destroy_session(self, session: ctypes.c_void_p) -> None:
        self._check(self._lib.fs_session_destroy(session))

    # ── Source ───────────────────────────────────────────────────────────

    def set_source_file(self, session: ctypes.c_void_p, path: str) -> None:
        self._check(self._lib.fs_session_set_source_file(session, path.encode("utf-8")))

    def set_source_image(self, session: ctypes.c_void_p, image: np.ndarray) -> None:
        """Set source from a numpy array (H, W, 3) BGR uint8."""
        fs_img = self._numpy_to_fs_image(image)
        self._check(self._lib.fs_session_set_source(session, ctypes.byref(fs_img)))

    # ── Swap ─────────────────────────────────────────────────────────────

    def swap_image_file(
        self, session: ctypes.c_void_p, target_path: str, output_path: str
    ) -> None:
        self._check(
            self._lib.fs_swap_image_file(
                session,
                target_path.encode("utf-8"),
                output_path.encode("utf-8"),
            )
        )

    def swap_image(self, session: ctypes.c_void_p, target: np.ndarray) -> np.ndarray:
        """Swap face on a numpy image, return result."""
        h, w = target.shape[:2]
        ch = 3
        out_buf = np.empty((h, w, ch), dtype=np.uint8)

        fs_in = self._numpy_to_fs_image(target)
        fs_out = _FsImageMut()
        fs_out.data = out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        fs_out.width = w
        fs_out.height = h
        fs_out.stride = w * ch
        fs_out.pixel_format = FS_PIXEL_BGR

        self._check(
            self._lib.fs_swap_image(session, ctypes.byref(fs_in), ctypes.byref(fs_out))
        )
        return out_buf

    def swap_video(
        self,
        session: ctypes.c_void_p,
        input_path: str,
        output_path: str,
        progress_cb=None,
    ) -> None:
        cb_type = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
        c_cb = cb_type(lambda i, t, _: progress_cb(i, t)) if progress_cb else None

        self._check(
            self._lib.fs_swap_video(
                session,
                input_path.encode("utf-8"),
                output_path.encode("utf-8"),
                c_cb,
                None,
            )
        )

    # ── Profiling ────────────────────────────────────────────────────────

    def get_timings(self, session: ctypes.c_void_p) -> dict:
        t = _FsTimings()
        self._check(self._lib.fs_get_timings(session, ctypes.byref(t)))
        return {
            "detection_ms": t.detection_ms,
            "landmarks_ms": t.landmarks_ms,
            "alignment_ms": t.alignment_ms,
            "swap_ms": t.swap_ms,
            "blend_ms": t.blend_ms,
            "total_ms": t.total_ms,
            "num_faces": t.num_faces,
        }

    def get_avg_fps(self, session: ctypes.c_void_p) -> float:
        fps = ctypes.c_float()
        self._check(self._lib.fs_get_avg_fps(session, ctypes.byref(fps)))
        return float(fps.value)

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _numpy_to_fs_image(arr: np.ndarray) -> _FsImage:
        """Convert a numpy BGR uint8 array to an FsImage struct."""
        assert arr.dtype == np.uint8, "Image must be uint8"
        assert arr.ndim == 3, "Image must be (H, W, C)"
        h, w, ch = arr.shape
        img = _FsImage()
        img.data = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        img.width = w
        img.height = h
        img.stride = arr.strides[0]
        img.pixel_format = FS_PIXEL_BGR
        return img

    def _check(self, err: int) -> None:
        if err != FS_OK:
            msg = self._lib.fs_error_string(err)
            raise RuntimeError(
                f"Native Face Swap error ({err}): {msg.decode('utf-8') if msg else 'unknown'}"
            )

    def _find_library(self) -> str:
        """Auto-discover the native library."""
        native_dir = Path(__file__).parent
        candidates = []

        if platform.system() == "Windows":
            candidates = list(native_dir.rglob("face_swap.dll"))
            candidates += list(native_dir.rglob("build/*/face_swap.dll"))
        else:
            candidates = list(native_dir.rglob("libface_swap.so"))
            candidates += list(native_dir.rglob("build/*/libface_swap.so"))

        if candidates:
            return str(candidates[0])

        raise FileNotFoundError(
            "Could not find native Face Swap library. "
            "Build it first with: cd face_swap/native && cmake -B build && cmake --build build"
        )

    def _setup_signatures(self) -> None:
        """Define ctypes function signatures for type safety."""
        L = self._lib

        L.fs_init.restype = ctypes.c_int
        L.fs_init.argtypes = []

        L.fs_shutdown.restype = ctypes.c_int
        L.fs_shutdown.argtypes = []

        L.fs_version.restype = ctypes.c_char_p
        L.fs_version.argtypes = []

        L.fs_error_string.restype = ctypes.c_char_p
        L.fs_error_string.argtypes = [ctypes.c_int]

        L.fs_config_default.restype = ctypes.c_int
        L.fs_config_default.argtypes = [ctypes.POINTER(_FsConfig)]

        L.fs_session_create.restype = ctypes.c_int
        L.fs_session_create.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_FsConfig),
        ]

        L.fs_session_destroy.restype = ctypes.c_int
        L.fs_session_destroy.argtypes = [ctypes.c_void_p]

        L.fs_session_set_source_file.restype = ctypes.c_int
        L.fs_session_set_source_file.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        L.fs_session_set_source.restype = ctypes.c_int
        L.fs_session_set_source.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FsImage)]

        L.fs_swap_image_file.restype = ctypes.c_int
        L.fs_swap_image_file.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
        ]

        L.fs_swap_image.restype = ctypes.c_int
        L.fs_swap_image.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_FsImage),
            ctypes.POINTER(_FsImageMut),
        ]

        L.fs_get_timings.restype = ctypes.c_int
        L.fs_get_timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(_FsTimings)]

        L.fs_get_avg_fps.restype = ctypes.c_int
        L.fs_get_avg_fps.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

    # ── Context manager ──────────────────────────────────────────────────

    def __enter__(self):
        self.init()
        return self

    def __exit__(self, *_exc):
        self.shutdown()
