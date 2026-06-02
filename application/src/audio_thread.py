"""
Audio thread for the live tab.

Captures microphone, runs VAD frame-by-frame, feeds Vosk in parallel,
and on speech-end runs SER + NLP. Updates the Pipeline.
"""
import json
import collections
import numpy as np
import sounddevice as sd
import torch
import tensorflow as tf
from PySide6.QtCore import QThread, Signal

from src import config
from src import models_loader
from src import audio_utils


class AudioThread(QThread):
    """Microphone + VAD + STT + SER + NLP.

    Signals:
      partial_transcript_changed(str): live word-by-word transcript
      utterance_finished(dict): when a full utterance is processed
        dict has keys: 'transcript', 'duration', 'ser_probs', 'nlp_probs'
    """
    partial_transcript_changed = Signal(str)
    utterance_finished = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline, lang='en', parent=None):
        super().__init__(parent)
        self._pipeline = pipeline
        self._lang = lang
        self._running = False
        self._stream = None
        self._audio_q = collections.deque()
        self._rec = None  # KaldiRecognizer

    def set_language(self, lang):
        """Change language. Call only while thread is stopped."""
        if self.isRunning():
            raise RuntimeError('Stop the thread before changing language')
        self._lang = lang

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f'[audio status] {status}')
        self._audio_q.append(indata.copy().squeeze())

    def run(self):
        try:
            self._rec = models_loader.make_recognizer(self._lang)
        except Exception as e:
            self.error.emit(f'Failed to create recognizer: {e}')
            return

        self._running = True

        try:
            self._stream = sd.InputStream(
                samplerate=config.SR, channels=1, dtype='float32',
                blocksize=config.VAD_CHUNK, callback=self._audio_callback)
            self._stream.start()
        except Exception as e:
            self.error.emit(f'Failed to open mic: {e}')
            self._running = False
            return

        # State machine
        state = 'SILENT'
        speech_buffer = []
        silent_run = 0
        last_partial = ''
        buf = np.empty(0, dtype=np.float32)

        try:
            while self._running:
                # Pull enough samples for one VAD chunk
                while len(buf) < config.VAD_CHUNK and self._running:
                    if self._audio_q:
                        buf = np.concatenate([buf, self._audio_q.popleft()])
                    else:
                        self.msleep(5)
                if not self._running:
                    break

                chunk = buf[:config.VAD_CHUNK]
                buf = buf[config.VAD_CHUNK:]

                # Vosk
                self._rec.AcceptWaveform(audio_utils.float_to_pcm16_bytes(chunk))

                # VAD
                with torch.no_grad():
                    prob = models_loader.vad_model(
                        torch.from_numpy(chunk), config.SR).item()

                if state == 'SILENT':
                    if prob >= config.SPEECH_START_THRESH:
                        state = 'SPEECH'
                        speech_buffer = [chunk]
                        silent_run = 0
                        last_partial = ''
                else:  # SPEECH
                    speech_buffer.append(chunk)

                    partial_obj = json.loads(self._rec.PartialResult())
                    partial = partial_obj.get('partial', '').strip()
                    if partial and partial != last_partial:
                        self._pipeline.update_partial(partial)
                        self.partial_transcript_changed.emit(partial)
                        last_partial = partial

                    if prob < config.SPEECH_END_THRESH:
                        silent_run += 1
                    else:
                        silent_run = 0

                    ulen = len(speech_buffer)
                    silence_threshold = (config.SILENCE_CHUNKS_AFTER_LONG
                                         if ulen >= config.LONG_SPEECH_THRESHOLD
                                         else config.SILENCE_CHUNKS_TO_END)

                    if (silent_run >= silence_threshold or
                            ulen >= config.MAX_SPEECH_CHUNKS):
                        # Utterance ended
                        audio = np.concatenate(speech_buffer).astype(np.float32)
                        duration = len(audio) / config.SR
                        state = 'SILENT'
                        speech_buffer = []
                        silent_run = 0

                        final_text = json.loads(
                            self._rec.FinalResult()).get('text', '').strip()
                        self._pipeline.clear_partial()

                        if ulen < config.MIN_SPEECH_CHUNKS:
                            # Too short, ignore but signal so UI can clear
                            self.utterance_finished.emit({
                                'transcript': '', 'duration': duration,
                                'ser_probs': None, 'nlp_probs': None,
                                'too_short': True,
                            })
                            continue

                        # SER
                        ser_probs = None
                        mel = audio_utils.prepare_for_ser(audio)
                        if mel is not None:
                            ser_probs = models_loader.ser_model.predict(
                                mel[np.newaxis, ...], verbose=0)[0]
                            self._pipeline.update_ser(ser_probs, duration=duration)

                        # NLP
                        nlp_probs = None
                        if final_text:
                            nlp_probs = models_loader.nlp_model(
                                tf.constant([final_text]), training=False).numpy()[0]
                            self._pipeline.update_nlp(nlp_probs, transcript=final_text)

                        self.utterance_finished.emit({
                            'transcript': final_text,
                            'duration': duration,
                            'ser_probs': ser_probs,
                            'nlp_probs': nlp_probs,
                            'too_short': False,
                        })
        finally:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()

    def stop(self):
        self._running = False
        self.wait(3000)
