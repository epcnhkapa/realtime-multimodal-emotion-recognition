"""
Face detection helper using MediaPipe Tasks API.

Uses the full-range BlazeFace model so faces 2-5m away are still detected
(important for movie scenes). Provides a helper to crop a tighter ROI
for the Vision emotion classifier (FER+ models prefer tightly cropped faces).
"""
import os
import urllib.request
import cv2

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import mediapipe as mp


MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'blaze_face_full_range.tflite')
MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/'
             'face_detector/blaze_face_full_range/float16/latest/'
             'blaze_face_full_range.tflite')


def _ensure_model():
    if os.path.isfile(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f'[face_detection] downloading model to {MODEL_PATH}...')
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print('[face_detection] model ready.')


def tighten_box(box, frame_shape, shrink_ratio=0.06):
    """Shrink a face bounding box inward so the ROI fed to the Vision model
    contains less hair/chin/ears and more pure face. Helps match the
    distribution of FER+ training data.

    Args:
      box: (x, y, w, h)
      frame_shape: (H, W) of the source frame
      shrink_ratio: fraction to crop from each side (0.06 = 6% off the sides)
    Returns:
      (x, y, w, h) clipped to frame
    """
    x, y, w, h = box
    H, W = frame_shape[:2]
    sx = int(w * shrink_ratio)
    sy = int(h * shrink_ratio * 0.4)  # less crop from top/bottom
    nx = max(0, x + sx)
    ny = max(0, y + sy)
    nw = max(1, w - 2 * sx)
    nh = max(1, h - 2 * sy)
    nw = min(nw, W - nx)
    nh = min(nh, H - ny)
    return (nx, ny, nw, nh)


class FaceDetector:
    """Per-thread face detector. MediaPipe is not safe to share across threads."""

    def __init__(self, min_detection_confidence=0.4):
        _ensure_model()
        base = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        opts = mp_vision.FaceDetectorOptions(
            base_options=base,
            min_detection_confidence=min_detection_confidence,
        )
        self._detector = mp_vision.FaceDetector.create_from_options(opts)

    def detect_largest(self, frame_bgr):
        """Run detection on a BGR frame, return the largest face box
        as (x, y, w, h) in pixel coords, or None."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        if not result.detections:
            return None

        h, w = frame_bgr.shape[:2]
        best_box = None
        best_area = 0
        for det in result.detections:
            bbox = det.bounding_box  # pixel coords
            x = max(0, int(bbox.origin_x))
            y = max(0, int(bbox.origin_y))
            bw = min(int(bbox.width), w - x)
            bh = min(int(bbox.height), h - y)
            if bw <= 0 or bh <= 0:
                continue
            area = bw * bh
            if area > best_area:
                best_area = area
                best_box = (x, y, bw, bh)
        return best_box

    def close(self):
        try:
            self._detector.close()
        except Exception:
            pass
