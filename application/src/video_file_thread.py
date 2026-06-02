"""
Audio analysis thread for the video file tab.
Plays nicely with QMediaPlayer: reads WAV at the player's pace.
"""
import os
import json
import wave
import tempfile
import subprocess
import numpy as np
import torch
import tensorflow as tf
from PySide6.QtCore import QThread, Signal

from src import config
from src import models_loader
from src import audio_utils


def _ffmpeg_exe():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def extract_audio_to_wav(video_path, target_sr=16000):
    fd, wav_path = tempfile.mkstemp(prefix='emo_', suffix='.wav')
    os.close(fd)
    cmd = [
        _ffmpeg_exe(),
        '-y', '-i', video_path,
        '-ac', '1', '-ar', str(target_sr),
        '-vn', '-loglevel', 'error',
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        try:
            os.remove(wav_path)
        except OSError:
            pass
        raise RuntimeError(
            f'ffmpeg failed: {result.stderr.decode("utf-8", errors="replace")}')
    return wav_path


# Lookahead: it's fine if we run up to 1s ahead of the player, then we sleep.
# (Otherwise we wake up every chunk, which is wasteful and chops VAD context.)
LOOKAHEAD_SAMPLES = int(1.0 * config.SR)
# Only treat large jumps as seeks. Small drifts are normal pacing.
SEEK_FORWARD_THRESHOLD = int(2.0 * config.SR)   # 2 seconds
SEEK_BACKWARD_THRESHOLD = int(2.0 * config.SR)  # 2 seconds


class AudioFileThread(QThread):
    partial_transcript_changed = Signal(str)
    utterance_finished = Signal(dict)
    error = Signal(str)
    finished_playing = Signal()

    def __init__(self, video_path, pipeline, get_position_ms,
                 lang='en', parent=None):
        super().__init__(parent)
        self._video_path = video_path
        self._pipeline = pipeline
        self._get_position = get_position_ms
        self._lang = lang
        self._running = False

    def run(self):
        try:
            wav_path = extract_audio_to_wav(self._video_path,
                                            target_sr=config.SR)
        except Exception as e:
            self.error.emit(f'Audio extract failed: {e}')
            return

        try:
            self._stream_wav(wav_path)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass
            self.finished_playing.emit()

    def _stream_wav(self, wav_path):
        try:
            rec = models_loader.make_recognizer(self._lang)
        except Exception as e:
            self.error.emit(f'Recognizer create failed: {e}')
            return

        try:
            wf = wave.open(wav_path, 'rb')
        except Exception as e:
            self.error.emit(f'Wave open failed: {e}')
            return

        sr = wf.getframerate()
        nch = wf.getnchannels()
        total_frames = wf.getnframes()

        if sr != config.SR or nch != 1:
            wf.close()
            self.error.emit(f'Unexpected WAV format: sr={sr} ch={nch}')
            return

        self._running = True

        state = 'SILENT'
        speech_buffer = []
        silent_run = 0
        last_partial = ''
        cursor = 0

        try:
            while self._running:
                pos_ms = self._get_position()
                target = int((pos_ms / 1000.0) * config.SR)

                # Real backward seek (user dragged slider back)
                if cursor > target + SEEK_BACKWARD_THRESHOLD:
                    cursor = max(0, target)
                    wf.setpos(cursor)
                    rec = models_loader.make_recognizer(self._lang)
                    state = 'SILENT'
                    speech_buffer = []
                    silent_run = 0
                    last_partial = ''
                    self._pipeline.clear_partial()
                    continue

                # Real forward seek (user dragged slider far ahead)
                if target > cursor + SEEK_FORWARD_THRESHOLD:
                    cursor = target
                    wf.setpos(cursor)
                    rec = models_loader.make_recognizer(self._lang)
                    state = 'SILENT'
                    speech_buffer = []
                    silent_run = 0
                    last_partial = ''
                    self._pipeline.clear_partial()
                    continue

                # If we've gotten ahead of the player by our lookahead, wait.
                # This is the natural backpressure that paces us with the video.
                if cursor >= target + LOOKAHEAD_SAMPLES:
                    self.msleep(30)
                    continue

                if cursor >= total_frames:
                    break

                raw = wf.readframes(config.VAD_CHUNK)
                if not raw:
                    break
                pcm = np.frombuffer(raw, dtype=np.int16)
                if len(pcm) < config.VAD_CHUNK:
                    pcm = np.pad(pcm, (0, config.VAD_CHUNK - len(pcm)))
                chunk = pcm.astype(np.float32) / 32768.0
                cursor += config.VAD_CHUNK

                rec.AcceptWaveform(audio_utils.float_to_pcm16_bytes(chunk))

                with torch.no_grad():
                    prob = models_loader.vad_model(
                        torch.from_numpy(chunk), config.SR).item()

                if state == 'SILENT':
                    if prob >= config.SPEECH_START_THRESH:
                        state = 'SPEECH'
                        speech_buffer = [chunk]
                        silent_run = 0
                        last_partial = ''
                else:
                    speech_buffer.append(chunk)

                    partial = json.loads(rec.PartialResult()).get(
                        'partial', '').strip()
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

                    if (silent_run >= silence_threshold
                            or ulen >= config.MAX_SPEECH_CHUNKS):
                        audio = np.concatenate(speech_buffer).astype(np.float32)
                        duration = len(audio) / config.SR
                        state = 'SILENT'
                        speech_buffer = []
                        silent_run = 0

                        final_text = json.loads(
                            rec.FinalResult()).get('text', '').strip()
                        self._pipeline.clear_partial()

                        if ulen < config.MIN_SPEECH_CHUNKS:
                            self.utterance_finished.emit({
                                'transcript': '', 'duration': duration,
                                'ser_probs': None, 'nlp_probs': None,
                                'too_short': True})
                            continue

                        ser_probs = None
                        mel = audio_utils.prepare_for_ser(audio)
                        if mel is not None:
                            ser_probs = models_loader.ser_model.predict(
                                mel[np.newaxis, ...], verbose=0)[0]
                            self._pipeline.update_ser(ser_probs, duration=duration)

                        nlp_probs = None
                        if final_text:
                            nlp_probs = models_loader.nlp_model(
                                tf.constant([final_text]),
                                training=False).numpy()[0]
                            self._pipeline.update_nlp(
                                nlp_probs, transcript=final_text)

                        self.utterance_finished.emit({
                            'transcript': final_text, 'duration': duration,
                            'ser_probs': ser_probs, 'nlp_probs': nlp_probs,
                            'too_short': False})
        finally:
            wf.close()

    def stop(self):
        self._running = False
        self.wait(3000)
