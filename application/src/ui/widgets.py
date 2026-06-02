"""
Reusable Qt widgets. Bounded sizes so layout doesn't reflow on text changes.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget, QLabel, QVBoxLayout, QHBoxLayout, QProgressBar, QFrame,
    QSizePolicy
)


EMOTION_COLORS = {
    'Angry':    '#e74c3c',
    'Fear':     '#9b59b6',
    'Happy':    '#f1c40f',
    'Neutral':  '#95a5a6',
    'Sad':      '#3498db',
    'Surprise': '#e67e22',
    'Negative': '#e74c3c',
    'Positive': '#2ecc71',
    '_default': '#7f8c8d',
}


def emotion_color(label):
    return EMOTION_COLORS.get(label, EMOTION_COLORS['_default'])


class EmotionBarWidget(QWidget):
    """Title + label + value + colored bar. Fixed height."""

    FIXED_HEIGHT = 44

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._title_text = title
        self._build_ui()
        self.setFixedHeight(self.FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.set_value(None, None)

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 4, 6, 4)
        v.setSpacing(2)

        head = QHBoxLayout()
        head.setSpacing(8)
        self.title_lbl = QLabel(self._title_text)
        title_font = QFont()
        title_font.setBold(True)
        self.title_lbl.setFont(title_font)
        head.addWidget(self.title_lbl)
        head.addStretch()

        self.label_lbl = QLabel('—')
        head.addWidget(self.label_lbl)

        self.value_lbl = QLabel('')
        self.value_lbl.setMinimumWidth(45)
        self.value_lbl.setAlignment(Qt.AlignRight)
        head.addWidget(self.value_lbl)
        v.addLayout(head)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        v.addWidget(self.bar)

    def set_value(self, label, confidence):
        if label is None or confidence is None:
            self.label_lbl.setText('—')
            self.value_lbl.setText('')
            self.bar.setValue(0)
            self.bar.setStyleSheet('')
            return
        self.label_lbl.setText(label)
        self.value_lbl.setText(f'{confidence:.2f}')
        self.bar.setValue(int(confidence * 100))
        color = emotion_color(label)
        self.bar.setStyleSheet(
            f'QProgressBar {{ background-color: #2b2b2b; border: 1px solid #444; '
            f'border-radius: 3px; }} '
            f'QProgressBar::chunk {{ background-color: {color}; }}')


class TopFusionWidget(QWidget):
    """Big top label. Fixed height."""

    FIXED_HEIGHT = 110

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._build_ui()
        self.set_fusion(None, None)

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(20, 14, 20, 14)
        v.setSpacing(2)

        self.label = QLabel('—')
        big = QFont()
        big.setPointSize(36)
        big.setBold(True)
        self.label.setFont(big)
        self.label.setAlignment(Qt.AlignCenter)
        v.addWidget(self.label)

        self.conf = QLabel('')
        small = QFont()
        small.setPointSize(11)
        self.conf.setFont(small)
        self.conf.setAlignment(Qt.AlignCenter)
        v.addWidget(self.conf)

    def set_fusion(self, emotion, confidence):
        if emotion is None or confidence is None:
            self.label.setText('—')
            self.conf.setText('waiting for input...')
            self.label.setStyleSheet('color: #888;')
            return
        self.label.setText(emotion.upper())
        self.conf.setText(f'confidence {confidence:.2f}')
        color = emotion_color(emotion)
        self.label.setStyleSheet(f'color: {color};')


class TranscriptWidget(QWidget):
    """Live (partial) transcript only — single line, larger font, centered.
    Final transcript is shown in FinalTranscriptWidget on the right panel.
    """

    FIXED_HEIGHT = 50

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.FIXED_HEIGHT)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(0)

        self.partial_lbl = QLabel('')
        f = QFont()
        f.setPointSize(13)
        f.setItalic(True)
        self.partial_lbl.setFont(f)
        self.partial_lbl.setStyleSheet('color: #bbb;')
        self.partial_lbl.setAlignment(Qt.AlignCenter)
        self.partial_lbl.setWordWrap(False)
        self.partial_lbl.setTextFormat(Qt.PlainText)
        v.addWidget(self.partial_lbl)

    def _elide(self, lbl, text):
        if not text:
            lbl.setText('')
            return
        fm = lbl.fontMetrics()
        width = max(50, lbl.width() - 10)
        elided = fm.elidedText(text, Qt.ElideRight, width)
        lbl.setText(elided)

    def set_partial(self, text):
        self._elide(self.partial_lbl, f'~ {text}' if text else '')

    def clear_partial(self):
        self.partial_lbl.setText('')


class FinalTranscriptWidget(QWidget):
    """Shows the most recent finalized utterance. Lives on the right panel
    under the per-modality bars. Wraps to multiple lines for long sentences.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)
        v.setContentsMargins(6, 6, 6, 6)
        v.setSpacing(4)

        head = QLabel('Last utterance')
        f = QFont()
        f.setBold(True)
        f.setPointSize(9)
        head.setFont(f)
        head.setStyleSheet('color: #888;')
        v.addWidget(head)

        self.text_lbl = QLabel('—')
        body = QFont()
        body.setPointSize(11)
        self.text_lbl.setFont(body)
        self.text_lbl.setStyleSheet('color: #ddd;')
        self.text_lbl.setWordWrap(True)
        self.text_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.text_lbl.setTextFormat(Qt.PlainText)
        v.addWidget(self.text_lbl, stretch=1)

    def set_text(self, text):
        if text:
            self.text_lbl.setText(f'"{text}"')
            self.text_lbl.setStyleSheet('color: #ddd;')
        else:
            self.text_lbl.setText('—')
            self.text_lbl.setStyleSheet('color: #666;')

    def clear(self):
        self.set_text('')


class VideoDisplayLabel(QLabel):
    """A QLabel for showing webcam/video frames."""

    def __init__(self, placeholder='', parent=None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(False)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(
            'background-color: #1a1a1a; color: #888; border: 1px solid #333;')


def make_separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Sunken)
    line.setFixedHeight(2)
    return line
