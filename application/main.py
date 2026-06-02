"""
Multimodal Emotion Recognition — application entry point.

Run from project root:
    py -3.13 -X utf8 main.py
or simply double-click run.bat
"""
import os
# Force UTF-8 BEFORE any TF/Keras import (TextVectorization with Turkish vocab)
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import sys
import traceback

from PySide6.QtWidgets import QApplication, QMessageBox

from src import models_loader
from src.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')   # consistent look across platforms

    # Heavy work: load all models BEFORE showing the main window so threads
    # can hit the loaded modules without races.
    try:
        models_loader.load_all(verbose=True)
    except Exception as e:
        traceback.print_exc()
        QMessageBox.critical(None, 'Model load failed',
                             f'Failed to load models:\n\n{e}\n\n'
                             f'Run from project root and ensure all model files exist.')
        return 1

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == '__main__':
    sys.exit(main())
