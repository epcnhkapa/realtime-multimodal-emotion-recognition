"""
NLP sentiment test: type a sentence, get sentiment prediction.
Ctrl+C to quit.

Run from project root:
  py -3.13 tests/nlp_test.py
"""
import os
# Force UTF-8 for Turkish vocabulary in the model
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import sys
import time
import numpy as np
import tensorflow as tf

MODEL_PATH = 'models/mixed_nlp.keras'
CLASSES = ['Negative', 'Neutral', 'Positive']

if not os.path.isfile(MODEL_PATH):
    raise SystemExit(f'NLP model not found at "{MODEL_PATH}". cwd={os.getcwd()}')

print('Loading NLP model...')
t0 = time.time()
model = tf.keras.models.load_model(MODEL_PATH)
print(f'Loaded in {time.time()-t0:.2f}s\n')


def predict(text):
    t0 = time.time()
    inp = tf.constant([text])
    probs = model(inp, training=False).numpy()[0]
    elapsed = (time.time() - t0) * 1000
    return probs, elapsed


TEST_SENTENCES = [
    "I am so happy today",
    "This is the worst thing ever",
    "The meeting is at three o'clock",
    "I love you so much",
    "I hate this",
    "It is okay I guess",
    "Bugun cok mutluyum",
    "Bu berbat bir durum",
    "Seni cok seviyorum",
]

print('=== Preset sentences ===\n')
for s in TEST_SENTENCES:
    probs, ms = predict(s)
    top = int(np.argmax(probs))
    print(f'"{s}"')
    print(f'  -> {CLASSES[top]:10s} ({probs[top]:.3f})  [{ms:.1f}ms]')
    print(f'     Neg={probs[0]:.2f}  Neu={probs[1]:.2f}  Pos={probs[2]:.2f}\n')


print('=== Interactive mode ===')
print('Type a sentence and press Enter. Ctrl+C to quit.\n')
try:
    while True:
        text = input('> ').strip()
        if not text:
            continue
        probs, ms = predict(text)
        top = int(np.argmax(probs))
        print(f'  -> {CLASSES[top]:10s} ({probs[top]:.3f})  [{ms:.1f}ms]')
        print(f'     Neg={probs[0]:.2f}  Neu={probs[1]:.2f}  Pos={probs[2]:.2f}\n')
except KeyboardInterrupt:
    print('\nBye.')
