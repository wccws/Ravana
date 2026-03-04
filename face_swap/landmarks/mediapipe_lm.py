"""
MediaPipe Face Mesh landmark detector.

As per PRD Section 5.4, this provides 468-point dense facial landmarks
for robust face alignment.
"""

from typing import List

import cv2
import numpy as np

from ..core.types import FaceBBox, Frame, Landmarks
from .base import LandmarkDetector


class MediaPipeLandmarkDetector(LandmarkDetector):
    """
    MediaPipe Face Mesh detector providing 468 dense landmarks.

    Features:
    - 468 facial landmarks for detailed face mesh
    - Handles expressions, moderate occlusions
    - Real-time performance
    """

    def __init__(
        self,
        device: str = "cpu",  # MediaPipe runs on CPU by default
        static_image_mode: bool = False,
        max_num_faces: int = 10,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe Face Mesh detector.

        Args:
            device: Device (MediaPipe primarily uses CPU)
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for detection
            min_tracking_confidence: Minimum confidence for tracking
        """
        super().__init__(device)
        self.static_image_mode = static_image_mode
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._face_mesh = None

    def load_model(self) -> None:
        """Load the MediaPipe Face Mesh model."""
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe is required. Install with: pip install mediapipe"
            )

        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            static_image_mode=self.static_image_mode,
            max_num_faces=self.max_num_faces,
            refine_landmarks=self.refine_landmarks,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence,
        )

    def detect(self, frame: Frame, bbox: FaceBBox) -> Landmarks:
        """
        Detect landmarks for a single face.

        Note: MediaPipe works on the full frame, so we crop to the bbox
        for efficiency and then map back to original coordinates.
        """
        if self._face_mesh is None:
            self.load_model()

        # Crop to bbox region (with margin for better detection)
        margin = 0.2
        x1 = int(max(0, bbox.x1 - bbox.width * margin))
        y1 = int(max(0, bbox.y1 - bbox.height * margin))
        x2 = int(min(frame.shape[1], bbox.x2 + bbox.width * margin))
        y2 = int(min(frame.shape[0], bbox.y2 + bbox.height * margin))

        crop = frame[y1:y2, x1:x2]

        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Process
        results = self._face_mesh.process(crop_rgb)

        if not results.multi_face_landmarks:
            # Return empty landmarks
            return Landmarks(points=np.zeros((468, 2), dtype=np.float32))

        # Get the first face's landmarks
        face_landmarks = results.multi_face_landmarks[0]

        # Convert to numpy array and map back to original coordinates
        h, w = crop.shape[:2]
        points = np.array(
            [[lm.x * w + x1, lm.y * h + y1] for lm in face_landmarks.landmark],
            dtype=np.float32,
        )

        return Landmarks(points=points, confidence=bbox.confidence)

    def detect_multi(self, frame: Frame, bboxes: List[FaceBBox]) -> List[Landmarks]:
        """Detect landmarks for multiple faces."""
        if self._face_mesh is None:
            self.load_model()

        # For multiple faces, process the full frame once
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = self._face_mesh.process(frame_rgb)

        landmarks_list = []

        if not results.multi_face_landmarks:
            # Return empty landmarks for all bboxes
            return [
                Landmarks(points=np.zeros((468, 2), dtype=np.float32)) for _ in bboxes
            ]

        # Match detected faces to bboxes based on proximity
        detected_faces = []
        for face_landmarks in results.multi_face_landmarks:
            points = np.array(
                [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
                dtype=np.float32,
            )

            # Calculate bbox center
            center = points.mean(axis=0)
            detected_faces.append((center, points))

        # Match each input bbox to closest detected face
        for bbox in bboxes:
            bbox_center = np.array([bbox.center.x, bbox.center.y])

            best_match = None
            best_distance = float("inf")

            for center, points in detected_faces:
                distance = np.linalg.norm(center - bbox_center)
                if distance < best_distance:
                    best_distance = distance
                    best_match = points

            if best_match is not None and best_distance < bbox.width:
                landmarks_list.append(
                    Landmarks(points=best_match, confidence=bbox.confidence)
                )
            else:
                landmarks_list.append(
                    Landmarks(points=np.zeros((468, 2), dtype=np.float32))
                )

        return landmarks_list

    def detect_full_frame(self, frame: Frame) -> List[Landmarks]:
        """
        Detect landmarks for all faces in a frame.

        Args:
            frame: Input image/frame

        Returns:
            List of Landmarks for all detected faces
        """
        if self._face_mesh is None:
            self.load_model()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        results = self._face_mesh.process(frame_rgb)

        landmarks_list = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = np.array(
                    [[lm.x * w, lm.y * h] for lm in face_landmarks.landmark],
                    dtype=np.float32,
                )

                landmarks_list.append(Landmarks(points=points))

        return landmarks_list
