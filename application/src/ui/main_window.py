"""
Main application window. Hosts Live tab and Video file tab.
"""
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QMainWindow, QTabWidget

from src.ui.live_tab import LiveTab
from src.ui.video_tab import VideoTab


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Multimodal Emotion Recognition')
        self.resize(1200, 800)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.live_tab = LiveTab()
        self.tabs.addTab(self.live_tab, 'Live')

        self.video_tab = VideoTab()
        self.tabs.addTab(self.video_tab, 'Video File')

    def closeEvent(self, event: QCloseEvent):
        try:
            self.live_tab.shutdown()
        except Exception as e:
            print(f'[MainWindow] error shutting down live tab: {e}')
        try:
            self.video_tab.shutdown()
        except Exception as e:
            print(f'[MainWindow] error shutting down video tab: {e}')
        event.accept()
