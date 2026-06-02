"""
Live vision test: webcam -> face detection -> emotion prediction
Press Q to quit.
"""
import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'vision_ferplus.keras')
CLASSES = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
IMG_SIZE = 48

print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)
print(f'Model loaded. Input shape: {model.input_shape}')

# Haar cascade for face detection (ships with opencv)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Cannot open webcam')

print('Webcam open. Press Q to quit.')

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(80, 80))

    # Use the largest face if multiple detected
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        roi = roi.astype(np.float32) / 255.0
        roi = roi.reshape(1, IMG_SIZE, IMG_SIZE, 1)

        probs = model.predict(roi, verbose=0)[0]
        top_idx = int(np.argmax(probs))
        top_label = CLASSES[top_idx]
        top_conf = float(probs[top_idx])

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{top_label} {top_conf:.2f}',
                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Print all probs on the side
        for i, (cls, p) in enumerate(zip(CLASSES, probs)):
            bar_width = int(p * 200)
            y_bar = 30 + i * 25
            cv2.rectangle(frame, (10, y_bar), (10 + bar_width, y_bar + 18),
                          (0, 200, 255), -1)
            cv2.putText(frame, f'{cls}: {p:.2f}',
                        (220, y_bar + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow('Vision test (press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print('Done.')
