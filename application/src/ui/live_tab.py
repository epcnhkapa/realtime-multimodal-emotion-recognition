"""
Tab 1: Live demo. Webcam + mic + multimodal emotion display.
"""
import numpy as np
import cv2
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QHBoxLayout, QVBoxLayout, QPushButton,
    QComboBox, QGroupBox, QSizePolicy
)

from src import config
from src.pipeline import Pipeline, top_emotion
from src.video_thread import VideoThread
from src.audio_thread import AudioThread
from src.ui.widgets import (
    EmotionBarWidget, TopFusionWidget, TranscriptWidget,
    FinalTranscriptWidget, VideoDisplayLabel, make_separator,
)


class LiveTab(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pipeline = Pipeline(name='live')
        self.video_thread = None
        self.audio_thread = None
        self._latest_frame = None

        self._build_ui()

        self.fusion_timer = QTimer(self)
        self.fusion_timer.setInterval(config.FUSION_UPDATE_INTERVAL_MS)
        self.fusion_timer.timeout.connect(self._refresh_fusion_display)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        self.top_fusion = TopFusionWidget()
        root.addWidget(self.top_fusion)

        root.addWidget(make_separator())

        # Middle row: video + transcript on the left (vertical stack);
        # bars + final transcript on the right (vertical stack).
        middle = QHBoxLayout()
        middle.setSpacing(12)

        # ---- left column: video on top, partial transcript directly below
        left_col = QVBoxLayout()
        left_col.setSpacing(4)
        self.video_label = VideoDisplayLabel('Webcam off — press Start')
        left_col.addWidget(self.video_label, stretch=1)
        self.transcript = TranscriptWidget()
        left_col.addWidget(self.transcript)
        left_wrapper = QWidget()
        left_wrapper.setLayout(left_col)
        middle.addWidget(left_wrapper, stretch=2)

        # ---- right column: per-modality bars, then final transcript
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

        root.addWidget(make_separator())

        footer = QHBoxLayout()
        footer.setSpacing(6)
        self.start_btn = QPushButton('Start')
        self.start_btn.clicked.connect(self.start_capture)
        footer.addWidget(self.start_btn)

        self.stop_btn = QPushButton('Stop')
        self.stop_btn.clicked.connect(self.stop_capture)
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

    # ---------------- control ----------------

    @Slot()
    def start_capture(self):
        if self.video_thread is not None or self.audio_thread is not None:
            return

        lang = self.lang_combo.currentData()
        self.lang_combo.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_lbl.setText(f'Running ({lang})')

        self.pipeline.reset()
        self.final_transcript.clear()

        self.video_thread = VideoThread(self.pipeline)
        self.video_thread.frame_ready.connect(self._on_frame_ready)
        self.video_thread.start()

        self.audio_thread = AudioThread(self.pipeline, lang=lang)
        self.audio_thread.partial_transcript_changed.connect(self._on_partial)
        self.audio_thread.utterance_finished.connect(self._on_utterance)
        self.audio_thread.error.connect(self._on_audio_error)
        self.audio_thread.start()

        self.fusion_timer.start()

    @Slot()
    def stop_capture(self):
        self.fusion_timer.stop()

        if self.audio_thread is not None:
            self.audio_thread.stop()
            self.audio_thread = None
        if self.video_thread is not None:
            self.video_thread.stop()
            self.video_thread = None

        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.lang_combo.setEnabled(True)
        self.status_lbl.setText('Stopped')
        self.video_label.setText('Webcam off — press Start')
        self.video_label.setPixmap(QPixmap())
        self._latest_frame = None

    def shutdown(self):
        self.stop_capture()

    # ---------------- video ----------------

    def _draw_pixmap(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg.copy())
        scaled = pix.scaled(self.video_label.size(),
                            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    @Slot(np.ndarray)
    def _on_frame_ready(self, frame_bgr):
        self._latest_frame = frame_bgr
        self._draw_pixmap(frame_bgr)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._latest_frame is not None:
            self._draw_pixmap(self._latest_frame)
        snap = self.pipeline.snapshot()
        self.transcript.set_partial(snap['partial_transcript'])

    # ---------------- audio events ----------------

    @Slot(str)
    def _on_partial(self, text):
        self.transcript.set_partial(text)

    @Slot(dict)
    def _on_utterance(self, info):
        self.transcript.clear_partial()
        if info.get('too_short'):
            self.status_lbl.setText(
                f'Running — last utterance ignored ({info["duration"]:.1f}s)')
            return
        self.final_transcript.set_text(info['transcript'])
        self.status_lbl.setText(
            f'Running — last utterance {info["duration"]:.1f}s')

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
