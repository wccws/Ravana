"""
Face alignment and cropping module.

As per PRD Section 5.4, this normalizes faces (rotation, scale)
and crops aligned face regions to standard sizes expected by the swap model.
"""

from typing import Tuple

import cv2
import numpy as np

from ..core.types import AlignedFace, FaceBBox, Frame, Landmarks


class FaceAligner:
    """
    Face aligner that crops and normalizes faces.

    Handles:
    - Rotation correction (eye line horizontal)
    - Scaling to standard crop sizes
    - Affine transformation for seamless blending back
    """

    # Standard face template for 256x256 crops
    # Points are: left eye, right eye, nose, left mouth, right mouth
    TEMPLATE_256 = np.array(
        [
            [89.5, 102.0],  # Left eye
            [166.5, 102.0],  # Right eye
            [128.0, 154.0],  # Nose
            [102.0, 205.0],  # Left mouth
            [154.0, 205.0],  # Right mouth
        ],
        dtype=np.float32,
    )

    # Standard template for 512x512 crops
    TEMPLATE_512 = np.array(
        [
            [179.0, 204.0],  # Left eye
            [333.0, 204.0],  # Right eye
            [256.0, 308.0],  # Nose
            [204.0, 410.0],  # Left mouth
            [308.0, 410.0],  # Right mouth
        ],
        dtype=np.float32,
    )

    def __init__(self, crop_size: Tuple[int, int] = (256, 256)):
        """
        Initialize face aligner.

        Args:
            crop_size: Target crop size (width, height)
        """
        self.crop_size = crop_size

        # Select appropriate template
        if crop_size == (256, 256):
            self.template = self.TEMPLATE_256
        elif crop_size == (512, 512):
            self.template = self.TEMPLATE_512
        else:
            # Scale template to requested size
            scale = crop_size[0] / 256.0
            self.template = self.TEMPLATE_256 * scale

    def align(
        self,
        frame: Frame,
        landmarks: Landmarks,
        bbox: FaceBBox,
        scale_factor: float = 1.0,
    ) -> AlignedFace:
        """
        Align and crop a face from a frame.

        Args:
            frame: Source image/frame
            landmarks: Facial landmarks
            bbox: Face bounding box
            scale_factor: Additional scaling factor

        Returns:
            AlignedFace with cropped image and transformation matrix
        """
        # Get reference points from landmarks
        src_points = self._get_reference_points(landmarks)

        # Destination points from template
        dst_points = self.template.copy()

        # Apply scale factor
        if scale_factor != 1.0:
            center = dst_points.mean(axis=0)
            dst_points = center + (dst_points - center) * scale_factor

        # Estimate affine transformation
        transform_matrix = cv2.estimateAffinePartial2D(
            src_points, dst_points, method=cv2.LMEDS
        )[0]

        # Apply transformation to crop face
        aligned_image = cv2.warpAffine(
            frame,
            transform_matrix,
            self.crop_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        return AlignedFace(
            image=aligned_image,
            transformation_matrix=transform_matrix,
            original_bbox=bbox,
            landmarks=landmarks,
            crop_size=self.crop_size,
        )

    def align_simple(
        self, frame: Frame, bbox: FaceBBox, scale_factor: float = 1.3
    ) -> AlignedFace:
        """
        Simple alignment based on bounding box (no rotation correction).

        Args:
            frame: Source image/frame
            bbox: Face bounding box
            scale_factor: Scale factor to expand the crop region

        Returns:
            AlignedFace with cropped image
        """
        # Expand bbox
        expanded_bbox = bbox.scale(scale_factor)

        x1 = int(max(0, expanded_bbox.x1))
        y1 = int(max(0, expanded_bbox.y1))
        x2 = int(min(frame.shape[1], expanded_bbox.x2))
        y2 = int(min(frame.shape[0], expanded_bbox.y2))

        # Crop
        crop = frame[y1:y2, x1:x2]

        # Resize to target size
        aligned_image = cv2.resize(crop, self.crop_size)

        # Create transformation matrix for inverse mapping
        # Maps from aligned coords back to original frame
        scale_x = (x2 - x1) / self.crop_size[0]
        scale_y = (y2 - y1) / self.crop_size[1]

        transform_matrix = np.array(
            [[scale_x, 0, x1], [0, scale_y, y1]], dtype=np.float32
        )

        return AlignedFace(
            image=aligned_image,
            transformation_matrix=transform_matrix,
            original_bbox=bbox,
            landmarks=None,
            crop_size=self.crop_size,
        )

    def get_inverse_transform(self, aligned_face: AlignedFace) -> np.ndarray:
        """
        Get inverse transformation matrix to map from aligned back to original.

        Args:
            aligned_face: Aligned face with transformation matrix

        Returns:
            Inverse transformation matrix
        """
        # Invert the 2x3 affine matrix
        transform_3x3 = np.vstack([aligned_face.transformation_matrix, [0, 0, 1]])

        inverse_3x3 = np.linalg.inv(transform_3x3)
        return inverse_3x3[:2, :]

    def _get_reference_points(self, landmarks: Landmarks) -> np.ndarray:
        """
        Extract reference points (eyes, nose, mouth) from landmarks.

        Args:
            landmarks: Facial landmarks

        Returns:
            Array of 5 reference points
        """
        points = landmarks.points

        if landmarks.num_points >= 68:
            # Standard 68-point landmarks
            left_eye = points[36:42].mean(axis=0)
            right_eye = points[42:48].mean(axis=0)
            nose_tip = points[33]
            left_mouth = points[48]
            right_mouth = points[54]

        elif landmarks.num_points == 468:
            # MediaPipe Face Mesh
            # Key landmark indices
            left_eye = points[468] if len(points) > 468 else points[33]
            right_eye = points[473] if len(points) > 473 else points[263]
            nose_tip = points[4]
            left_mouth = points[61]
            right_mouth = points[291]

        else:
            # Fallback: estimate from available points
            # Assume evenly distributed points
            left_eye = points[landmarks.num_points // 4]
            right_eye = points[landmarks.num_points // 2]
            nose_tip = points[landmarks.num_points * 3 // 4]
            left_mouth = points[landmarks.num_points * 4 // 5]
            right_mouth = points[landmarks.num_points * 9 // 10]

        return np.array(
            [left_eye, right_eye, nose_tip, left_mouth, right_mouth], dtype=np.float32
        )


def get_face_aligner(crop_size: Tuple[int, int] = (256, 256)) -> FaceAligner:
    """
    Factory function to get a face aligner.

    Args:
        crop_size: Target crop size

    Returns:
        FaceAligner instance
    """
    return FaceAligner(crop_size=crop_size)
