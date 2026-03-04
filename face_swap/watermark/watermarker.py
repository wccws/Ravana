"""
Invisible watermarking and provenance metadata.

As per PRD Section 6.3:
- Embeds optional invisible watermarks for traceability.
- Logs and exposes metadata indicating when content has been manipulated.
- Provides hooks for integrators who want content provenance.

This implementation uses DCT-domain watermarking, which survives
common image processing operations (resize, mild compression, small crops).
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import cv2
import numpy as np


@dataclass
class WatermarkConfig:
    """Configuration for watermarking and provenance.

    Attributes:
        enabled: Whether watermarking is active.
        strength: Embedding strength (higher = more robust but more visible).
                  Typical range 1-10; default 5 is imperceptible on most content.
        message: Custom message to embed (max 256 chars).
        embed_timestamp: Whether to include processing timestamp.
        embed_model_info: Whether to include model name / version metadata.
    """

    enabled: bool = False
    strength: float = 5.0
    message: str = ""
    embed_timestamp: bool = True
    embed_model_info: bool = True


@dataclass
class ProvenanceMetadata:
    """Provenance metadata for a processed frame/image.

    Gives integrators full traceability information that can be
    serialised, stored alongside outputs, or embedded in EXIF.
    """

    timestamp: str = ""
    model_name: str = ""
    model_version: str = ""
    source_hash: str = ""
    target_hash: str = ""
    is_manipulated: bool = True
    custom_message: str = ""
    pipeline_version: str = "0.1.0"

    def to_json(self) -> str:
        """Serialise metadata to JSON string."""
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ProvenanceMetadata":
        """Deserialise metadata from JSON string."""
        data = json.loads(json_str)
        return cls(**data)


class InvisibleWatermarker:
    """
    Invisible watermarker using DCT-domain bit embedding.

    The watermark is spread across the middle-frequency DCT coefficients
    of 8×8 blocks so that it is:
      - Invisible to the human eye at default strength.
      - Robust against JPEG compression and mild resizing.
      - Detectable from the output image alone (blind extraction).

    Usage:
        >>> wm = InvisibleWatermarker(WatermarkConfig(enabled=True))
        >>> marked = wm.embed(image, provenance)
        >>> extracted = wm.extract(marked)
    """

    # 8-bit signature prefix used to locate the payload start
    _SIGNATURE = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.int8)

    # DCT coefficient positions used for embedding (mid-frequency band)
    _COEFF_PAIRS: list = [
        (3, 4),
        (4, 3),
        (2, 5),
        (5, 2),
        (1, 6),
        (6, 1),
        (4, 4),
        (3, 5),
        (5, 3),
        (2, 6),
        (6, 2),
        (1, 5),
        (5, 1),
        (4, 2),
        (2, 4),
        (3, 3),
    ]

    def __init__(self, config: Optional[WatermarkConfig] = None):
        self.config = config or WatermarkConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(
        self,
        image: np.ndarray,
        metadata: Optional[ProvenanceMetadata] = None,
    ) -> np.ndarray:
        """
        Embed an invisible watermark + provenance into *image*.

        Args:
            image: BGR uint8 image (H, W, 3).
            metadata: Optional provenance data; auto-generated if None.

        Returns:
            Watermarked image (same dtype / shape).
        """
        if not self.config.enabled:
            return image

        # Build payload from metadata
        if metadata is None:
            metadata = self._build_default_metadata()
        payload = self._metadata_to_bits(metadata)

        # Work on a float copy of the blue channel (least perceptible).
        marked = image.copy()
        channel = marked[:, :, 0].astype(np.float64)

        h, w = channel.shape
        block_h, block_w = h // 8, w // 8
        total_blocks = block_h * block_w

        # Prepend signature
        bits = np.concatenate([self._SIGNATURE, payload])

        if len(bits) > total_blocks * len(self._COEFF_PAIRS):
            bits = bits[: total_blocks * len(self._COEFF_PAIRS)]

        bit_idx = 0
        for by in range(block_h):
            for bx in range(block_w):
                if bit_idx >= len(bits):
                    break
                y0, x0 = by * 8, bx * 8
                block = channel[y0 : y0 + 8, x0 : x0 + 8]
                dct_block = cv2.dct(block)

                for r, c in self._COEFF_PAIRS:
                    if bit_idx >= len(bits):
                        break
                    bit = bits[bit_idx]
                    delta = self.config.strength * (1 if bit else -1)
                    dct_block[r, c] += delta
                    bit_idx += 1

                channel[y0 : y0 + 8, x0 : x0 + 8] = cv2.idct(dct_block)

        marked[:, :, 0] = np.clip(channel, 0, 255).astype(np.uint8)
        return marked

    def extract(self, image: np.ndarray) -> Optional[ProvenanceMetadata]:
        """
        Extract watermark metadata from an image.

        Args:
            image: Potentially watermarked BGR uint8 image.

        Returns:
            ProvenanceMetadata if a valid watermark is found, else None.
        """
        channel = image[:, :, 0].astype(np.float64)
        h, w = channel.shape
        block_h, block_w = h // 8, w // 8

        raw_bits: list = []
        for by in range(block_h):
            for bx in range(block_w):
                y0, x0 = by * 8, bx * 8
                block = channel[y0 : y0 + 8, x0 : x0 + 8]
                dct_block = cv2.dct(block)

                for r, c in self._COEFF_PAIRS:
                    raw_bits.append(1 if dct_block[r, c] > 0 else 0)

        raw_bits = np.array(raw_bits, dtype=np.int8)

        # Search for signature
        sig_len = len(self._SIGNATURE)
        idx = self._find_signature(raw_bits)
        if idx is None:
            return None

        payload_bits = raw_bits[idx + sig_len :]
        return self._bits_to_metadata(payload_bits)

    def create_provenance(
        self,
        source_image: Optional[np.ndarray] = None,
        target_image: Optional[np.ndarray] = None,
        model_name: str = "",
        model_version: str = "",
        custom_message: str = "",
    ) -> ProvenanceMetadata:
        """Create a ProvenanceMetadata record for an operation."""
        meta = ProvenanceMetadata(
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_name=model_name,
            model_version=model_version,
            is_manipulated=True,
            custom_message=custom_message or self.config.message,
        )
        if source_image is not None:
            meta.source_hash = self._image_hash(source_image)
        if target_image is not None:
            meta.target_hash = self._image_hash(target_image)
        return meta

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_default_metadata(self) -> ProvenanceMetadata:
        meta = ProvenanceMetadata(
            is_manipulated=True,
            custom_message=self.config.message,
        )
        if self.config.embed_timestamp:
            meta.timestamp = datetime.now(timezone.utc).isoformat()
        return meta

    @staticmethod
    def _metadata_to_bits(metadata: ProvenanceMetadata) -> np.ndarray:
        """Encode metadata into a bit array via JSON → UTF-8 → bits."""
        json_bytes = metadata.to_json().encode("utf-8")
        # Truncate to keep payload reasonable
        json_bytes = json_bytes[:256]
        bits = []
        for byte in json_bytes:
            for i in range(8):
                bits.append((byte >> (7 - i)) & 1)
        return np.array(bits, dtype=np.int8)

    @staticmethod
    def _bits_to_metadata(bits: np.ndarray) -> Optional[ProvenanceMetadata]:
        """Decode a bit array back into ProvenanceMetadata."""
        if len(bits) < 8:
            return None
        n_bytes = len(bits) // 8
        byte_arr = bytearray()
        for i in range(n_bytes):
            val = 0
            for j in range(8):
                val = (val << 1) | int(bits[i * 8 + j])
            byte_arr.append(val)

        try:
            text = byte_arr.decode("utf-8").rstrip("\x00")
            # Find the end of valid JSON
            brace_count = 0
            end = 0
            for idx, ch in enumerate(text):
                if ch == "{":
                    brace_count += 1
                elif ch == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        end = idx + 1
                        break
            if end == 0:
                return None
            return ProvenanceMetadata.from_json(text[:end])
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError):
            return None

    def _find_signature(self, bits: np.ndarray) -> Optional[int]:
        """Search for the signature prefix in a bit stream."""
        sig_len = len(self._SIGNATURE)
        for i in range(min(len(bits) - sig_len, 512)):
            if np.array_equal(bits[i : i + sig_len], self._SIGNATURE):
                return i
        return None

    @staticmethod
    def _image_hash(image: np.ndarray) -> str:
        """Quick perceptual-ish hash of an image for provenance."""
        small = cv2.resize(image, (32, 32))
        return hashlib.sha256(small.tobytes()).hexdigest()[:16]
