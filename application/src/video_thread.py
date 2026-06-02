"""
Video capture thread for the live tab.

Uses MediaPipe for face detection (much more robust than Haar Cascade).
The detector instance lives inside this thread because MediaPipe is not
safe to share across threads.
"""
import time
import numpy as np
import cv2
from PySide6.QtCore import QThread, Signal

from src import config
from src import models_loader
from src.face_detection import FaceDetector, tighten_box

class VideoThread(QThread):
    """Webcam thread.

    Emits:
      frame_ready(np.ndarray BGR): a frame ready to draw in the UI
    """
    frame_ready = Signal(np.ndarray)

    def __init__(self, pipeline, parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._running = False
        self._cap = None

    def run(self):
        self._cap = cv2.VideoCapture(config.WEBCAM_DEVICE_INDEX)
        if not self._cap.isOpened():
            print('[VideoThread] cannot open webcam')
            return

        # MediaPipe detector belongs to this thread
        face_detector = FaceDetector(min_detection_confidence=0.5)

        self._running = True
        last_predict = 0.0

        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    self.msleep(10)
                    continue

                now = time.time()
                if now - last_predict >= config.VIDEO_PREDICT_INTERVAL:
                    box = face_detector.detect_largest(frame)
                    if box is not None:
                        # Tighter crop for the Vision model (better matches FER+ data)
                        tx, ty, tw, th = tighten_box(box, frame.shape)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        roi = gray[ty:ty+th, tx:tx+tw]
                        if roi.size > 0:
                            roi = cv2.resize(roi, (config.VISION_IMG_SIZE,
                                                   config.VISION_IMG_SIZE))
                            roi = roi.astype(np.float32) / 255.0
                            roi = roi.reshape(1, config.VISION_IMG_SIZE,
                                              config.VISION_IMG_SIZE, 1)
                            probs = models_loader.vision_model.predict(
                                roi, verbose=0)[0]
                            # Original box stays for display (user sees the
                            # full detection); Vision uses the tighter crop
                            # Show the tighter box on screen so the user sees
                            # exactly what the Vision model is looking at
                            self._pipeline.update_vision(probs, face_box=(tx, ty, tw, th))
                    else:
                        # No face: keep last vision_probs but clear box
                        self._pipeline.update_vision(
                            self._pipeline.snapshot()['vision_probs'],
                            face_box=None)
                    last_predict = now

                # Draw the face box and current top vision label
                display_frame = frame.copy()
                snap = self._pipeline.snapshot()
                if (snap['vision_face_box'] is not None
                        and snap['vision_probs'] is not None):
                    bx, by, bw, bh = snap['vision_face_box']
                    top_idx = int(np.argmax(snap['vision_probs']))
                    top_label = config.EMOTIONS[top_idx]
                    top_conf = float(snap['vision_probs'][top_idx])
                    cv2.rectangle(display_frame, (bx, by), (bx+bw, by+bh),
                                  (0, 200, 0), 2)
                    cv2.putText(display_frame, f'{top_label} {top_conf:.2f}',
                                (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 200, 0), 2)

                self.frame_ready.emit(display_frame)
                self.msleep(15)
        finally:
            face_detector.close()
            if self._cap is not None:
                self._cap.release()

    def stop(self):
        self._running = False
        self.wait(2000)
