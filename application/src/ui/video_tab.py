"""
Tab 2: Video file emotion analysis.

Audio path:  QMediaPlayer -> QAudioOutput -> speakers (Qt handles sync)
Video path:  QMediaPlayer -> QVideoSink -> our handler -> draw box -> QLabel
"""
import os
import time
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, QUrl, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtMultimedia import (
    QMediaPlayer, QAudioOutput, QVideoFrame, QVideoSink
)
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QFileDialog, QSlider, QSizePolicy
)

from src import config
from src.pipeline import Pipeline, top_emotion
from src.video_file_thread import AudioFileThread
from src import models_loader
from src.face_detection import FaceDetector, tighten_box
from src.ui.widgets import (
    EmotionBarWidget, TopFusionWidget, TranscriptWidget,
    FinalTranscriptWidget, VideoDisplayLabel,
    make_separator, emotion_color,
)


def _hex_to_bgr(hex_color):
    h = hex_color.lstrip('#')
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (b, g, r)


class VideoTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = Pipeline(name='video')
        self.audio_thread = None
        self._video_path = None
        self._face_detector = None
        self._last_predict_time = 0.0
        self._user_seeking = False
        self._latest_bgr = None

        self._build_ui()

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(1.0)
        self.player.setAudioOutput(self.audio_output)

        self.video_sink = QVideoSink(self)
        self.player.setVideoSink(self.video_sink)
        self.video_sink.videoFrameChanged.connect(self._on_video_frame)

        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.errorOccurred.connect(self._on_player_error)

        self.fusion_timer = QTimer(self)
        self.fusion_timer.setInterval(config.FUSION_UPDATE_INTERVAL_MS)
        self.fusion_timer.timeout.connect(self._refresh_fusion_display)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # File picker row
        file_row = QHBoxLayout()
        self.load_btn = QPushButton('Load Video...')
        self.load_btn.clicked.connect(self.pick_video)
        file_row.addWidget(self.load_btn)
        self.path_lbl = QLabel('No video loaded')
        self.path_lbl.setStyleSheet('color: #888;')
        self.path_lbl.setTextFormat(Qt.PlainText)
        file_row.addWidget(self.path_lbl, stretch=1)
        file_widget = QWidget()
        file_widget.setLayout(file_row)
        file_widget.setFixedHeight(34)
        root.addWidget(file_widget)

        self.top_fusion = TopFusionWidget()
        root.addWidget(self.top_fusion)
        root.addWidget(make_separator())

        # Middle: video stack on left, bars + final transcript on right
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # left column: video + partial transcript directly under
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        self.video_label = VideoDisplayLabel('Load a video to begin')
        left_col.addWidget(self.video_label, stretch=1)
        self.transcript = TranscriptWidget()
        left_col.addWidget(self.transcript)
        left_wrapper = QWidget()
        left_wrapper.setLayout(left_col)
        middle.addWidget(left_wrapper, stretch=2)

        # right column: bars + final transcript
        right_col = QVBoxLayout()
        right_col.setSpacing(8)

        bars_container = QGroupBox('Per-modality')
        bars_container.setSizePolicy(QSizePolicy.Preferred,
                                     QSizePolicy.Preferred)
        bars_v = QVBoxLayout(bars_container)
        bars_v.setSpacing(6)
        self.vision_bar = EmotionBarWidget('Vision')
        self.ser_bar = EmotionBarWidget('SER')
        self.nlp_bar = EmotionBarWidget('NLP')
        bars_v.addWidget(self.vision_bar)
        bars_v.addWidget(self.ser_bar)
        bars_v.addWidget(self.nlp_bar)
        right_col.addWidget(bars_container)

        self.final_transcript = FinalTranscriptWidget()
        right_col.addWidget(self.final_transcript, stretch=1)

        right_wrapper = QWidget()
        right_wrapper.setLayout(right_col)
        middle.addWidget(right_wrapper, stretch=1)

        root.addLayout(middle, stretch=1)

        # Seek slider
        seek_row = QHBoxLayout()
        seek_row.setSpacing(8)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.sliderPressed.connect(self._on_seek_pressed)
        self.seek_slider.sliderReleased.connect(self._on_seek_released)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)
        seek_row.addWidget(self.seek_slider, stretch=1)
        self.time_lbl = QLabel('0:00 / 0:00')
        self.time_lbl.setStyleSheet('color: #aaa;')
        seek_row.addWidget(self.time_lbl)
        seek_widget = QWidget()
        seek_widget.setLayout(seek_row)
        seek_widget.setFixedHeight(28)
        root.addWidget(seek_widget)

        root.addWidget(make_separator())

        # Footer
        footer = QHBoxLayout()
        footer.setSpacing(6)
        self.play_btn = QPushButton('Play')
        self.play_btn.clicked.connect(self.play_video)
        self.play_btn.setEnabled(False)
        footer.addWidget(self.play_btn)

        self.pause_btn = QPushButton('Pause')
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        footer.addWidget(self.pause_btn)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_video)
        self.stop_btn.setEnabled(False)
        footer.addWidget(self.stop_btn)

        footer.addStretch()

        footer.addWidget(QLabel('Language:'))
        self.lang_combo = QComboBox()
        self.lang_combo.addItem('English', 'en')
        self.lang_combo.addItem('Türkçe', 'tr')
        if config.DEFAULT_LANG == 'tr':
            self.lang_combo.setCurrentIndex(1)
        footer.addWidget(self.lang_combo)

        self.status_lbl = QLabel('Idle')
        self.status_lbl.setStyleSheet('color: #888; padding-left: 12px;')
        footer.addWidget(self.status_lbl)

        footer_widget = QWidget()
        footer_widget.setLayout(footer)
        footer_widget.setFixedHeight(36)
        root.addWidget(footer_widget)

    # ---------------- file picker ----------------

    @Slot()
    def pick_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Pick a video file',
            os.path.abspath('assets/demo_videos'),
            'Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*.*)')
        if not path:
            return
        self._video_path = path
        self._set_path_label(path)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.play_btn.setEnabled(True)
        self.status_lbl.setText('Ready')

    def _set_path_label(self, path):
        fm = self.path_lbl.fontMetrics()
        width = max(200, self.path_lbl.width() - 10)
        self.path_lbl.setText(fm.elidedText(path, Qt.ElideMiddle, width))
        self.path_lbl.setStyleSheet('color: #ddd;')
        self.path_lbl.setToolTip(path)

    # ---------------- playback control ----------------

    @Slot()
    def play_video(self):
        if not self._video_path:
            return

        lang = self.lang_combo.currentData()
        self.lang_combo.setEnabled(False)
        self.load_btn.setEnabled(False)
        self.play_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText('Pause')
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText(f'Playing ({lang})')

        self.pipeline.reset()
        self.final_transcript.clear()
        self._face_detector = FaceDetector(min_detection_confidence=0.4)
        self._last_predict_time = 0.0
        self._latest_bgr = None

        self.audio_thread = AudioFileThread(
            self._video_path, self.pipeline,
            get_position_ms=lambda: self.player.position(),
            lang=lang)
        self.audio_thread.partial_transcript_changed.connect(self._on_partial)
        self.audio_thread.utterance_finished.connect(self._on_utterance)
        self.audio_thread.error.connect(self._on_audio_error)
        self.audio_thread.start()

        self.fusion_timer.start()
        self.player.play()

    @Slot()
    def toggle_pause(self):
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
            self.pause_btn.setText('Resume')
            self.status_lbl.setText('Paused')
        else:
            self.player.play()
            self.pause_btn.setText('Pause')
            self.status_lbl.setText('Playing')

    @Slot()
    def stop_video(self):
        self.fusion_timer.stop()
        self.player.stop()

        if self.audio_thread is not None:
            self.audio_thread.stop()
            self.audio_thread = None

        if self._face_detector is not None:
            self._face_detector.close()
            self._face_detector = None

        self.lang_combo.setEnabled(True)
        self.load_btn.setEnabled(True)
        self.play_btn.setEnabled(self._video_path is not None)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText('Pause')
        self.stop_btn.setEnabled(False)
        self.status_lbl.setText('Stopped')

        self.video_label.setPixmap(QPixmap())
        self.video_label.setText('Load a video to begin')
        self._latest_bgr = None

    @Slot(QMediaPlayer.Error, str)
    def _on_player_error(self, err, msg):
        if err != QMediaPlayer.NoError:
            self.status_lbl.setText(f'Player error: {msg}')

    def shutdown(self):
        self.stop_video()

    # ---------------- video frame interception ----------------

    @Slot(QVideoFrame)
    def _on_video_frame(self, frame):
        if not frame.isValid():
            return

        try:
            img = frame.toImage()
        except Exception:
            return
        if img.isNull():
            return

        img = img.convertToFormat(QImage.Format_RGB888)
        w = img.width()
        h = img.height()
        if w <= 0 or h <= 0:
            return

        ptr = img.constBits()
        bpl = img.bytesPerLine()
        buf = bytes(ptr)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, bpl)
        rgb = arr[:, :w*3].reshape(h, w, 3).copy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Face detection on every frame (cheap), Vision only on interval (heavier)
        current_box = None
        if self._face_detector is not None:
            box = self._face_detector.detect_largest(bgr)
            if box is not None:
                tx, ty, tw, th = tighten_box(box, bgr.shape)
                current_box = (tx, ty, tw, th)

        now = time.time()
        if (current_box is not None
                and now - self._last_predict_time >= config.VIDEO_PREDICT_INTERVAL):
            tx, ty, tw, th = current_box
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            roi = gray[ty:ty+th, tx:tx+tw]
            if roi.size > 0:
                roi = cv2.resize(roi, (config.VISION_IMG_SIZE,
                                       config.VISION_IMG_SIZE))
                roi = roi.astype(np.float32) / 255.0
                roi = roi.reshape(1, config.VISION_IMG_SIZE,
                                  config.VISION_IMG_SIZE, 1)
                probs = models_loader.vision_model.predict(roi, verbose=0)[0]
                self.pipeline.update_vision(probs, face_box=current_box)
            self._last_predict_time = now
        elif current_box is not None:
            self.pipeline.update_vision(
                self.pipeline.snapshot()['vision_probs'],
                face_box=current_box)
        elif current_box is None:
            self.pipeline.update_vision(
                self.pipeline.snapshot()['vision_probs'], face_box=None)

        snap = self.pipeline.snapshot()
        if (snap['vision_face_box'] is not None
                and snap['vision_probs'] is not None):
            bx, by, bw, bh = snap['vision_face_box']
            top_idx = int(np.argmax(snap['vision_probs']))
            top_label = config.EMOTIONS[top_idx]
            top_conf = float(snap['vision_probs'][top_idx])
            color = _hex_to_bgr(emotion_color(top_label))
            cv2.rectangle(bgr, (bx, by), (bx+bw, by+bh), color, 3)
            cv2.putText(bgr, f'{top_label} {top_conf:.2f}',
                        (bx, max(20, by - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self._latest_bgr = bgr
        self._draw_pixmap(bgr)

    def _draw_pixmap(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        scaled = pix.scaled(self.video_label.size(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    # ---------------- player signals ----------------

    @Slot('qint64')
    def _on_position_changed(self, pos_ms):
        if not self._user_seeking:
            self.seek_slider.setValue(int(pos_ms))
        dur = self.seek_slider.maximum()
        self.time_lbl.setText(f'{_fmt_time(pos_ms)} / {_fmt_time(dur)}')

    @Slot('qint64')
    def _on_duration_changed(self, dur_ms):
        self.seek_slider.setRange(0, int(dur_ms))
        self.time_lbl.setText(
            f'{_fmt_time(self.player.position())} / {_fmt_time(dur_ms)}')

    @Slot()
    def _on_seek_pressed(self):
        self._user_seeking = True

    @Slot()
    def _on_seek_released(self):
        self.player.setPosition(self.seek_slider.value())
        self._user_seeking = False

    @Slot(int)
    def _on_seek_moved(self, value):
        self.time_lbl.setText(
            f'{_fmt_time(value)} / {_fmt_time(self.seek_slider.maximum())}')

    # ---------------- audio thread signals ----------------

    @Slot(str)
    def _on_partial(self, text):
        self.transcript.set_partial(text)

    @Slot(dict)
    def _on_utterance(self, info):
        self.transcript.clear_partial()
        if not info.get('too_short'):
            self.final_transcript.set_text(info['transcript'])

    @Slot(str)
    def _on_audio_error(self, msg):
        self.status_lbl.setText(f'Audio error: {msg}')

    # ---------------- fusion display ----------------

    @Slot()
    def _refresh_fusion_display(self):
        snap = self.pipeline.snapshot()

        if snap['vision_probs'] is not None:
            label, conf = top_emotion(snap['vision_probs'])
            self.vision_bar.set_value(label, conf)
        if snap['ser_probs'] is not None:
            label, conf = top_emotion(snap['ser_probs'])
            self.ser_bar.set_value(label, conf)
        if snap['nlp_probs'] is not None:
            idx = int(np.argmax(snap['nlp_probs']))
            self.nlp_bar.set_value(
                config.NLP_CLASSES[idx], float(snap['nlp_probs'][idx]))

        out = self.pipeline.compute_fusion()
        if out is not None:
            fused, _weights = out
            label, conf = top_emotion(fused)
            self.top_fusion.set_fusion(label, conf)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._latest_bgr is not None:
            self._draw_pixmap(self._latest_bgr)
        snap = self.pipeline.snapshot()
        self.transcript.set_partial(snap['partial_transcript'])
        if self._video_path:
            self._set_path_label(self._video_path)


def _fmt_time(ms):
    s = max(0, int(ms / 1000))
    return f'{s // 60}:{s % 60:02d}'
