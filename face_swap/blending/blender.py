"""
Face blending and color correction module.

As per PRD Section 5.7, this seamlessly composites swapped faces back
into the original frame using Poisson blending or alpha blending with
color correction.
"""

import cv2
import numpy as np

from ..core.types import Frame, SwapResult


class FaceBlender:
    """
    Face blender that composites swapped faces into original frames.

    Supports:
    - Poisson blending (gradient-domain)
    - Alpha blending with color correction
    - Mask-based feathering
    """

    def __init__(self, blend_mode: str = "alpha", color_correction: bool = True):
        """
        Initialize face blender.

        Args:
            blend_mode: Blending mode ("poisson", "alpha", "feather")
            color_correction: Whether to apply color correction
        """
        self.blend_mode = blend_mode
        self.color_correction = color_correction

    def blend(
        self,
        frame: Frame,
        swap_result: SwapResult,
        feather_amount: int = 15,
    ) -> Frame:
        """
        Blend a swapped face into the original frame.

        Args:
            frame: Original frame
            swap_result: Swap result with swapped face, mask, and alignment info
            feather_amount: Amount of feathering for the blend

        Returns:
            Blended frame
        """
        swapped_face = swap_result.swapped_face
        mask = swap_result.mask
        aligned = swap_result.target_aligned

        # Get inverse transformation to map swapped face back to frame
        transform_inv = self._get_inverse_transform(aligned.transformation_matrix)

        # Warp swapped face back to original frame
        h, w = frame.shape[:2]
        warped_face = cv2.warpAffine(
            swapped_face,
            transform_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Warp mask back to original frame
        warped_mask = cv2.warpAffine(
            mask,
            transform_inv,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        # Apply color correction
        if self.color_correction:
            warped_face = self._color_correct(warped_face, frame, warped_mask)

        # Blend based on mode
        if self.blend_mode == "poisson":
            result = self._poisson_blend(frame, warped_face, warped_mask)
        elif self.blend_mode == "feather":
            result = self._feather_blend(
                frame, warped_face, warped_mask, feather_amount
            )
        else:  # alpha
            result = self._alpha_blend(frame, warped_face, warped_mask)

        return result

    def blend_multi(
        self,
        frame: Frame,
        swap_results: list,
    ) -> Frame:
        """
        Blend multiple swapped faces into a frame.

        Args:
            frame: Original frame
            swap_results: List of swap results

        Returns:
            Blended frame
        """
        result = frame.copy()

        for swap_result in swap_results:
            result = self.blend(result, swap_result)

        return result

    def _alpha_blend(
        self,
        frame: Frame,
        swapped_face: Frame,
        mask: np.ndarray,
    ) -> Frame:
        """
        Alpha blend using formula: final = mask * swapped + (1 - mask) * original

        As per PRD Section 5.7, this is the performant fallback.

        Args:
            frame: Original frame
            swapped_face: Warped swapped face
            mask: Blending mask (0-1)

        Returns:
            Blended frame
        """
        # Expand mask to 3 channels
        mask_3ch = np.stack([mask] * 3, axis=-1)

        # Alpha blend
        blended = (mask_3ch * swapped_face + (1 - mask_3ch) * frame).astype(np.uint8)

        return blended

    def _feather_blend(
        self,
        frame: Frame,
        swapped_face: Frame,
        mask: np.ndarray,
        feather_amount: int,
    ) -> Frame:
        """
        Feathered blend with expanded mask.

        Args:
            frame: Original frame
            swapped_face: Warped swapped face
            mask: Initial mask
            feather_amount: Feather radius

        Returns:
            Blended frame
        """
        # Dilate and blur mask for feathering
        kernel = np.ones((feather_amount, feather_amount), np.uint8)
        mask_dilated = cv2.dilate(mask, kernel, iterations=1)
        mask_feathered = cv2.GaussianBlur(
            mask_dilated, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0
        )

        # Clamp to [0, 1]
        mask_feathered = np.clip(mask_feathered, 0, 1)

        return self._alpha_blend(frame, swapped_face, mask_feathered)

    def _poisson_blend(
        self,
        frame: Frame,
        swapped_face: Frame,
        mask: np.ndarray,
    ) -> Frame:
        """
        Poisson blending (gradient-domain blending).

        As per PRD Section 5.7, this provides seamless composition
        by matching gradients.

        Note: This is a simplified implementation. Full Poisson blending
        can be computationally expensive for real-time use.

        Args:
            frame: Original frame
            swapped_face: Warped swapped face
            mask: Blending mask

        Returns:
            Blended frame
        """
        # For performance, use OpenCV's seamlessClone if available
        # Find center of mask
        mask_binary = (mask > 0.5).astype(np.uint8) * 255

        # Get bounding box of mask region
        coords = cv2.findNonZero(mask_binary)
        if coords is None:
            return frame

        x, y, w, h = cv2.boundingRect(coords)
        center = (x + w // 2, y + h // 2)

        # Crop region
        src_crop = swapped_face[y : y + h, x : x + w]
        mask_crop = mask_binary[y : y + h, x : x + w]

        # Check if mask is valid
        if mask_crop.sum() < 100:  # Too small
            return self._alpha_blend(frame, swapped_face, mask)

        try:
            # Use OpenCV seamless clone
            blended = cv2.seamlessClone(
                src_crop,
                frame,
                mask_crop,
                center,
                cv2.NORMAL_CLONE,  # or MIXED_CLONE, MONOCHROME_TRANSFER
            )
            return blended
        except cv2.error:
            # Fallback to alpha blend if seamlessClone fails
            return self._alpha_blend(frame, swapped_face, mask)

    def _color_correct(
        self,
        swapped_face: Frame,
        target_frame: Frame,
        mask: np.ndarray,
    ) -> Frame:
        """
        Apply local color correction to match surrounding region.

        As per PRD Section 5.7, this adjusts skin tone, brightness,
        and contrast to match the surrounding region.

        Args:
            swapped_face: Warped swapped face
            target_frame: Original target frame
            mask: Blending mask

        Returns:
            Color-corrected swapped face
        """
        # Expand mask slightly to include surrounding region
        kernel = np.ones((21, 21), np.uint8)
        mask_dilated = cv2.dilate((mask > 0.1).astype(np.uint8), kernel, iterations=2)

        # Calculate statistics in LAB color space (better for color matching)
        swapped_lab = cv2.cvtColor(swapped_face, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target_frame, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Get valid pixels
        mask_bool = mask_dilated > 0

        if mask_bool.sum() < 100:
            return swapped_face

        # Calculate mean and std for each channel
        result = swapped_lab.copy()

        for i in range(3):  # L, A, B channels
            swapped_chan = swapped_lab[:, :, i]
            target_chan = target_lab[:, :, i]

            # Calculate statistics
            swapped_mean = swapped_chan[mask_bool].mean()
            swapped_std = swapped_chan[mask_bool].std()
            target_mean = target_chan[mask_bool].mean()
            target_std = target_chan[mask_bool].std()

            if swapped_std > 0:
                # Match statistics
                result[:, :, i] = (swapped_chan - swapped_mean) * (
                    target_std / swapped_std
                ) + target_mean

        # Convert back to BGR
        result = np.clip(result, 0, 255).astype(np.uint8)
        result_bgr = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        return result_bgr

    def _get_inverse_transform(self, transform_matrix: np.ndarray) -> np.ndarray:
        """
        Get inverse affine transformation.

        Args:
            transform_matrix: 2x3 forward affine matrix

        Returns:
            2x3 inverse affine matrix
        """
        # Convert to 3x3 for inversion
        matrix_3x3 = np.vstack([transform_matrix, [0, 0, 1]])

        try:
            inverse_3x3 = np.linalg.inv(matrix_3x3)
            return inverse_3x3[:2, :]
        except np.linalg.LinAlgError:
            # Return identity if inversion fails
            return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)


def create_blender(
    blend_mode: str = "alpha", color_correction: bool = True
) -> FaceBlender:
    """
    Factory function to create a face blender.

    Args:
        blend_mode: Blending mode ("poisson", "alpha", "feather")
        color_correction: Whether to apply color correction

    Returns:
        FaceBlender instance
    """
    return FaceBlender(blend_mode=blend_mode, color_correction=color_correction)
