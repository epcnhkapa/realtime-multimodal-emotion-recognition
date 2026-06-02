"""
Pipeline state container + fusion logic.
One Pipeline instance per tab (live / video file). Threads update its
state via the `update_*` methods; UI reads via `snapshot()` and
`compute_fusion()`.
"""
import time
import threading
import numpy as np

from src import config


class Pipeline:
    """Holds per-tab state and computes the fused emotion vector.

    Thread safety: all state mutations + reads go through `_lock`.
    Read methods return copies (or read-only snapshots) so callers
    can use them without holding the lock.
    """

    def __init__(self, name='pipeline'):
        self.name = name
        self._lock = threading.Lock()

        # Vision: 6-d softmax probs over EMOTIONS, updated continuously
        # Stored value is EWMA-smoothed across recent frames so the UI
        # doesn't jitter between adjacent emotions.
        self._vision_probs = None
        self._vision_face_box = None

        # SER: 6-d softmax probs, updated on speech-end
        self._ser_probs = None
        self._ser_timestamp = 0.0        # when SER last updated

        # Vision: track when face was last seen for forget-after timeout
        self._vision_face_last_seen = 0.0

        # NLP: 3-d softmax probs (Neg/Neu/Pos), updated on speech-end
        self._nlp_probs = None
        self._nlp_timestamp = 0.0        # when NLP last updated

        # STT
        self._partial_transcript = ''
        self._final_transcript = ''
        self._transcripts_history = []   # list of (timestamp, text)

        # SER chunk metadata (for UI debugging)
        self._last_speech_duration = 0.0

    # ---------------- update methods ----------------

    def update_vision(self, probs, face_box=None):
        now = time.time()
        with self._lock:
            if face_box is not None and probs is not None:
                self._vision_face_last_seen = now
                # EWMA: smoothed = alpha * new + (1-alpha) * old
                if self._vision_probs is None:
                    self._vision_probs = probs.copy()
                else:
                    a = config.VISION_SMOOTHING_ALPHA
                    self._vision_probs = (a * probs +
                                          (1.0 - a) * self._vision_probs)
                self._vision_face_box = face_box
            else:
                # No face: forget vision if it's been too long
                if now - self._vision_face_last_seen >= config.VISION_FORGET_AFTER:
                    self._vision_probs = None
                self._vision_face_box = None

    def update_ser(self, probs, duration=None):
        now = time.time()
        with self._lock:
            self._ser_probs = probs
            self._ser_timestamp = now
            if duration is not None:
                self._last_speech_duration = duration

    def update_nlp(self, probs, transcript=''):
        now = time.time()
        with self._lock:
            self._nlp_probs = probs
            self._nlp_timestamp = now
            self._final_transcript = transcript
            if transcript:
                self._transcripts_history.append((now, transcript))
                # Keep only last 5
                self._transcripts_history = self._transcripts_history[-5:]

    def update_partial(self, partial):
        with self._lock:
            self._partial_transcript = partial

    def clear_partial(self):
        with self._lock:
            self._partial_transcript = ''

    def reset(self):
        """Reset all state. Used when switching language or restarting."""
        with self._lock:
            self._vision_probs = None
            self._vision_face_box = None
            self._vision_face_last_seen = 0.0
            self._ser_probs = None
            self._ser_timestamp = 0.0
            self._nlp_probs = None
            self._nlp_timestamp = 0.0
            self._partial_transcript = ''
            self._final_transcript = ''
            self._transcripts_history = []
            self._last_speech_duration = 0.0

    # ---------------- read methods ----------------

    def snapshot(self):
        """Return a dict with copies of current state. Safe to use without lock."""
        with self._lock:
            return {
                'vision_probs': None if self._vision_probs is None
                                else self._vision_probs.copy(),
                'vision_face_box': self._vision_face_box,
                'ser_probs': None if self._ser_probs is None
                             else self._ser_probs.copy(),
                'ser_timestamp': self._ser_timestamp,
                'nlp_probs': None if self._nlp_probs is None
                             else self._nlp_probs.copy(),
                'nlp_timestamp': self._nlp_timestamp,
                'partial_transcript': self._partial_transcript,
                'final_transcript': self._final_transcript,
                'transcripts_history': list(self._transcripts_history),
                'last_speech_duration': self._last_speech_duration,
            }

    def compute_fusion(self, now=None):
        """Compute the fused 6-d emotion vector at time `now` (default: time.time()).
        Returns (fused_vec, weights_tuple) or None if vision not available yet.
        weights_tuple = (w_vision, w_ser, w_nlp) after normalization.

        SER and NLP both decay exponentially since their last update so that
        old utterances stop dominating after long silences.
        """
        if now is None:
            now = time.time()
        s = self.snapshot()
        v = s['vision_probs']
        a = s['ser_probs']
        n = s['nlp_probs']
        ser_t = s['ser_timestamp']
        nlp_t = s['nlp_timestamp']

        if v is None:
            return None

        w_v = config.W_VISION

        # SER weight: full at the moment of speech-end, decays since
        w_a = 0.0
        if a is not None:
            dt_ser = now - ser_t
            w_a = config.W_SER * (0.5 ** (dt_ser / config.SER_HALF_LIFE))

        # NLP weight: only if confident, then decays
        w_n = 0.0
        if n is not None:
            n_conf = float(n.max())
            if n_conf >= config.NLP_CONF_THRESH:
                dt_nlp = now - nlp_t
                w_n = config.W_NLP_PEAK * (0.5 ** (dt_nlp / config.NLP_HALF_LIFE))

        total = w_v + w_a + w_n
        if total <= 0:
            return None
        w_v, w_a, w_n = w_v / total, w_a / total, w_n / total

        result = w_v * v
        if a is not None and w_a > 0:
            result = result + w_a * a
        if n is not None and w_n > 0:
            result = result + w_n * (n @ config.NLP_TO_EMOTION)

        s_sum = result.sum()
        if s_sum > 0:
            result = result / s_sum
        return result, (w_v, w_a, w_n)


def top_emotion(probs):
    """Helper: return (label, confidence) for a probability vector."""
    idx = int(np.argmax(probs))
    return config.EMOTIONS[idx], float(probs[idx])
