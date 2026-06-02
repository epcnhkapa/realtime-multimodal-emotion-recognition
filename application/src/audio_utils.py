"""
Audio preprocessing utilities. Pure functions, no state.
"""
import numpy as np
import librosa

from src import config


def fix_length(y, target_len=None):
    """Center-crop or zero-pad a 1-D waveform to `target_len` samples."""
    if target_len is None:
        target_len = config.SER_TARGET_LEN
    if len(y) > target_len:
        start = (len(y) - target_len) // 2
        return y[start:start + target_len]
    elif len(y) < target_len:
        return np.pad(y, (0, target_len - len(y)))
    return y


def waveform_to_logmel(y):
    """Compute normalized log-mel spectrogram (matches SER training pipeline)."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=config.SR,
        n_fft=config.SER_N_FFT,
        hop_length=config.SER_HOP,
        n_mels=config.SER_N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)


def trim_silence(y):
    """Remove leading/trailing silence (matches training)."""
    y_trim, _ = librosa.effects.trim(y, top_db=config.SER_TRIM_TOP_DB)
    return y_trim


def float_to_pcm16_bytes(chunk_f32):
    """Convert float32 [-1, 1] waveform to 16-bit PCM bytes for Vosk."""
    clipped = np.clip(chunk_f32, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    return pcm.tobytes()


def prepare_for_ser(audio):
    """Full SER preprocessing: trim -> length-fix -> log-mel -> add channel.
    Returns shape (N_MELS, T, 1) ready for batching, or None if too short.
    """
    y_trim = trim_silence(audio)
    if len(y_trim) < config.SR * 0.3:
        return None
    y_fixed = fix_length(y_trim)
    mel = waveform_to_logmel(y_fixed)
    return mel[..., np.newaxis]
