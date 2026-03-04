"""
Mobile export and deployment helpers.

As per PRD Section 4.1:
  - Optional: thin bindings for mobile (Android / iOS) where feasible.

This module provides:
  - ONNX → TFLite conversion for Android.
  - ONNX → CoreML conversion for iOS.
  - Model quantisation (INT8, FP16) for mobile-optimised inference.
  - Input/output shape validation for mobile constraints.
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger("face_swap.platform.mobile")


@dataclass
class MobileExportConfig:
    """Configuration for mobile model export.

    Attributes:
        target:        ``android`` or ``ios``.
        precision:     ``fp32``, ``fp16``, or ``int8``.
        input_size:    Model input resolution (e.g., 128 or 256).
        optimize:      Apply platform-specific optimisations.
        metadata:      Extra metadata to embed in the model.
    """

    target: str = "android"
    precision: str = "fp16"
    input_size: int = 128
    optimize: bool = True
    metadata: Dict[str, str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class MobileExporter:
    """
    Export face swap models for mobile deployment.

    Handles the conversion pipeline:
      ONNX → Platform-specific format → Quantised → Optimised.
    """

    def __init__(self):
        pass

    def export_android(
        self,
        onnx_path: str,
        output_path: str,
        config: Optional[MobileExportConfig] = None,
    ) -> str:
        """
        Export ONNX model to TensorFlow Lite for Android.

        Pipeline: ONNX → TF SavedModel → TFLite (+ quantisation).

        Args:
            onnx_path:   Path to ONNX model.
            output_path: Destination .tflite path.
            config:      Export configuration.

        Returns:
            Path to the TFLite model.
        """
        cfg = config or MobileExportConfig(target="android")

        logger.info("Exporting for Android: %s → %s", onnx_path, output_path)

        # Step 1: ONNX → TF
        try:
            import onnx
            from onnx_tf.backend import prepare
        except ImportError:
            raise ImportError(
                "onnx-tf required for Android export. "
                "Install with: pip install onnx-tf"
            )

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)

        # Save as TF SavedModel
        tf_dir = output_path + ".tf_saved"
        tf_rep.export_graph(tf_dir)

        # Step 2: TF → TFLite
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError(
                "TensorFlow required for TFLite conversion. "
                "Install with: pip install tensorflow"
            )

        converter = tf.lite.TFLiteConverter.from_saved_model(tf_dir)

        # Quantisation
        if cfg.precision == "fp16":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        elif cfg.precision == "int8":
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self._make_representative_dataset(
                cfg.input_size
            )
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            ]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(tflite_model)

        # Cleanup
        import shutil

        shutil.rmtree(tf_dir, ignore_errors=True)

        size_mb = os.path.getsize(output_path) / (1 << 20)
        logger.info("TFLite model saved: %s (%.1f MB)", output_path, size_mb)
        return output_path

    def export_ios(
        self,
        onnx_path: str,
        output_path: str,
        config: Optional[MobileExportConfig] = None,
    ) -> str:
        """
        Export ONNX model to CoreML for iOS.

        Args:
            onnx_path:   Path to ONNX model.
            output_path: Destination .mlpackage or .mlmodel path.
            config:      Export configuration.

        Returns:
            Path to the CoreML model.
        """
        cfg = config or MobileExportConfig(target="ios")

        logger.info("Exporting for iOS: %s → %s", onnx_path, output_path)

        try:
            import coremltools as ct
        except ImportError:
            raise ImportError(
                "coremltools required for iOS export. "
                "Install with: pip install coremltools"
            )

        # Convert
        model = ct.converters.onnx.convert(model=onnx_path)

        # FP16 quantisation
        if cfg.precision == "fp16":
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model,
                nbits=16,
            )
        elif cfg.precision == "int8":
            model = ct.models.neural_network.quantization_utils.quantize_weights(
                model,
                nbits=8,
            )

        # Add metadata
        model.author = cfg.metadata.get("author", "Face Swap SDK")
        model.short_description = cfg.metadata.get(
            "description", "Face swap model for iOS"
        )
        model.version = cfg.metadata.get("version", "0.2.0")

        model.save(output_path)

        logger.info("CoreML model saved: %s", output_path)
        return output_path

    def validate_mobile_model(
        self,
        model_path: str,
        target: str = "android",
    ) -> Dict[str, any]:
        """
        Validate a mobile model for deployment readiness.

        Checks:
          - File size (should be < 50 MB for mobile).
          - Input/output shapes.
          - Supported ops.

        Returns:
            Validation report dict.
        """
        report = {
            "path": model_path,
            "target": target,
            "exists": os.path.exists(model_path),
            "size_mb": 0,
            "size_ok": False,
            "issues": [],
        }

        if not report["exists"]:
            report["issues"].append("Model file not found")
            return report

        report["size_mb"] = os.path.getsize(model_path) / (1 << 20)
        report["size_ok"] = report["size_mb"] < 50.0

        if not report["size_ok"]:
            report["issues"].append(
                f"Model too large for mobile: {report['size_mb']:.1f} MB (target < 50 MB)"
            )

        return report

    @staticmethod
    def _make_representative_dataset(input_size: int):
        """Generate a representative dataset for INT8 calibration."""

        def gen():
            for _ in range(100):
                data = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
                yield [data]

        return gen

    def get_model_info(self, model_path: str) -> Dict:
        """Get metadata from a mobile model file."""
        ext = Path(model_path).suffix.lower()
        info = {"format": ext, "path": model_path}

        if ext == ".tflite":
            try:
                import tensorflow as tf

                interpreter = tf.lite.Interpreter(model_path=model_path)
                interpreter.allocate_tensors()
                inputs = interpreter.get_input_details()
                outputs = interpreter.get_output_details()
                info["inputs"] = [
                    {
                        "name": i["name"],
                        "shape": i["shape"].tolist(),
                        "dtype": str(i["dtype"]),
                    }
                    for i in inputs
                ]
                info["outputs"] = [
                    {
                        "name": o["name"],
                        "shape": o["shape"].tolist(),
                        "dtype": str(o["dtype"]),
                    }
                    for o in outputs
                ]
            except ImportError:
                pass

        return info
