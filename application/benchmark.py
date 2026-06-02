"""
Standalone latency + model size benchmark.

Run from project root, after the app already runs successfully:
    py -3.13 -X utf8 benchmark.py

What it measures (CPU only):
  - Vision CNN forward pass on a 48x48x1 input
  - Silero VAD forward pass on a 512-sample chunk
  - SER CNN+BiLSTM forward pass on a 128x188x1 log-mel
  - NLP text classifier forward pass on a short string
  - MediaPipe face detection on a 480p synthetic frame
  - Vosk STT full decode of a 3-second synthetic waveform

Reports mean, std, p50, p95 in milliseconds. Saves results to
benchmark_results.json so they can be pasted into the report.

Notes:
  - Warm-up shots are excluded from statistics (first inference is
    always slow due to lazy graph construction).
  - All measurements use time.perf_counter for monotonic high-resolution
    timing.
  - Random inputs are used so we measure raw inference cost, not
    end-to-end pipeline cost.
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence TF info logs

import sys
import time
import json
import platform
import statistics
from pathlib import Path

import numpy as np


# ----------------- helpers -----------------

def stats_ms(times_s):
    """Convert a list of seconds to ms stats dict."""
    arr = np.array(times_s) * 1000.0
    return {
        'n': int(len(arr)),
        'mean_ms': float(arr.mean()),
        'std_ms': float(arr.std()),
        'p50_ms': float(np.percentile(arr, 50)),
        'p95_ms': float(np.percentile(arr, 95)),
        'min_ms': float(arr.min()),
        'max_ms': float(arr.max()),
    }


def bench(name, fn, n_warmup=5, n_iter=100):
    """Run fn() n_warmup times (discarded), then n_iter times timed."""
    print(f'  [{name}] warmup ({n_warmup})...', end='', flush=True)
    for _ in range(n_warmup):
        fn()
    print(' measure...', end='', flush=True)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    s = stats_ms(times)
    print(f' mean={s["mean_ms"]:.2f}ms p95={s["p95_ms"]:.2f}ms')
    return s


def dir_size_mb(path):
    """Recursively size a directory in MB."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    total = 0
    for f in p.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ----------------- main -----------------

def main():
    results = {
        'system': {
            'python': sys.version.split()[0],
            'platform': platform.platform(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
        },
        'latency': {},
        'model_sizes_mb': {},
    }

    print(f'System: {results["system"]["platform"]}')
    print(f'CPU: {results["system"]["processor"]} ({results["system"]["cpu_count"]} cores)')
    print(f'Python: {results["system"]["python"]}')
    print()

    # --- Load models ---
    print('Loading models...')
    from src import models_loader, config
    models_loader.load_all(verbose=False)
    print('  done.')
    print()

    # --- Model sizes ---
    print('Model sizes:')
    sizes = {
        'vision_cnn': dir_size_mb(config.VISION_PATH),
        'ser_cnn_bilstm': dir_size_mb(config.SER_PATH),
        'nlp_classifier': dir_size_mb(config.NLP_PATH),
        'vosk_en': dir_size_mb(config.VOSK_PATHS['en']),
        'vosk_tr': dir_size_mb(config.VOSK_PATHS['tr']),
    }
    # Silero VAD is downloaded by silero_vad package, may not have a
    # static path. Try to find it via the package.
    try:
        import silero_vad
        vad_dir = Path(silero_vad.__file__).parent
        # silero_vad ships an onnx file ~1.7 MB
        vad_size = sum(f.stat().st_size for f in vad_dir.rglob('*.onnx'))
        sizes['silero_vad_onnx'] = vad_size / (1024 * 1024)
    except Exception:
        sizes['silero_vad_onnx'] = None

    for k, v in sizes.items():
        if v is None:
            print(f'  {k:20s}  (not found)')
        else:
            print(f'  {k:20s}  {v:7.2f} MB')

    valid_sizes = [v for v in sizes.values() if v is not None]
    print(f'  {"TOTAL":20s}  {sum(valid_sizes):7.2f} MB')
    results['model_sizes_mb'] = sizes
    results['model_sizes_mb']['_total'] = sum(valid_sizes)
    print()

    # --- Vision CNN ---
    print('Latency:')
    vision_in = np.random.rand(
        1, config.VISION_IMG_SIZE, config.VISION_IMG_SIZE, 1).astype(np.float32)
    results['latency']['vision_cnn'] = bench(
        'vision_cnn',
        lambda: models_loader.vision_model.predict(vision_in, verbose=0),
        n_warmup=5, n_iter=200)

    # --- SER ---
    ser_in = np.random.rand(1, config.SER_N_MELS, 188, 1).astype(np.float32)
    results['latency']['ser_cnn_bilstm'] = bench(
        'ser_cnn_bilstm',
        lambda: models_loader.ser_model.predict(ser_in, verbose=0),
        n_warmup=3, n_iter=100)

    # --- NLP ---
    import tensorflow as tf
    nlp_in = tf.constant(['this is a benchmark sentence to measure latency'])
    results['latency']['nlp_classifier'] = bench(
        'nlp_classifier',
        lambda: models_loader.nlp_model(nlp_in, training=False).numpy(),
        n_warmup=5, n_iter=200)

    # --- VAD ---
    import torch
    vad_chunk = torch.from_numpy(
        np.random.randn(config.VAD_CHUNK).astype(np.float32))
    def vad_call():
        with torch.no_grad():
            models_loader.vad_model(vad_chunk, config.SR)
    results['latency']['silero_vad_per_chunk'] = bench(
        'silero_vad_per_chunk',
        vad_call, n_warmup=10, n_iter=500)

    # --- MediaPipe face detection ---
    from src.face_detection import FaceDetector
    fd = FaceDetector(min_detection_confidence=0.5, model_selection=1)
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    results['latency']['face_detection_mediapipe'] = bench(
        'face_detection_mediapipe',
        lambda: fd.detect_largest(frame), n_warmup=5, n_iter=100)
    fd.close()

    # --- Vosk full decode of 3-second synthetic audio ---
    # Vosk doesn't expose a clean "single call", but we can feed 3s of
    # audio chunk-by-chunk and measure the total wall time.
    rec_en = models_loader.make_recognizer('en')
    fake_audio = (np.random.randn(3 * config.SR) * 0.05).astype(np.float32)
    pcm_chunks = []
    for i in range(0, len(fake_audio), config.VAD_CHUNK):
        chunk = fake_audio[i:i + config.VAD_CHUNK]
        if len(chunk) < config.VAD_CHUNK:
            break
        pcm_chunks.append((chunk * 32767).astype(np.int16).tobytes())

    def vosk_full_decode():
        rec = models_loader.make_recognizer('en')
        for pcm in pcm_chunks:
            rec.AcceptWaveform(pcm)
        rec.FinalResult()
    results['latency']['vosk_en_3s_full_decode'] = bench(
        'vosk_en_3s_full_decode',
        vosk_full_decode, n_warmup=2, n_iter=20)

    # --- Fusion tick ---
    # Measures only the late fusion compute_fusion() call, with all three
    # modalities populated. This is the per-tick cost of the UI timer.
    from src.pipeline import Pipeline
    pl = Pipeline('bench')
    pl.update_vision(np.random.rand(6).astype(np.float32) / 6,
                     face_box=(0, 0, 100, 100))
    pl.update_ser(np.random.rand(6).astype(np.float32) / 6)
    pl.update_nlp(np.random.rand(3).astype(np.float32) / 3, transcript='x')
    results['latency']['fusion_tick'] = bench(
        'fusion_tick',
        lambda: pl.compute_fusion(),
        n_warmup=10, n_iter=1000)

    # --- Save ---
    out_path = 'benchmark_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print()
    print(f'Saved: {out_path}')
    print()

    # --- Summary table for the report ---
    print('Summary (paste into report):')
    print(f'  {"Component":30s} {"mean (ms)":>10s} {"p95 (ms)":>10s}')
    print(f'  {"-"*30} {"-"*10} {"-"*10}')
    for k, v in results['latency'].items():
        print(f'  {k:30s} {v["mean_ms"]:>10.2f} {v["p95_ms"]:>10.2f}')


if __name__ == '__main__':
    main()
