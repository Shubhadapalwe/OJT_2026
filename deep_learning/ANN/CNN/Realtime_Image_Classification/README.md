# Real-Time CNN Image Classification — MobileNetV2 + Webcam

## Overview

This is the first complete project in the repository — a real-time image classifier that uses your laptop's webcam, processes each frame with a pre-trained MobileNetV2 CNN, and displays the top-1 prediction with confidence score as an overlay on the live video. No training required — this uses a model already trained on ImageNet (1.4 million images, 1000 classes).

---

## File in This Folder

| File | Topic |
|------|-------|
| `29_Marvellous_Realtime_Image_Classification.py` | Live webcam classification using MobileNetV2 + TensorFlow/Keras |

---

## What It Does

1. Loads the pre-trained **MobileNetV2** model (weights from ImageNet — downloads automatically on first run)
2. Opens the **webcam** via OpenCV
3. For each video frame: preprocesses → predicts → decodes → overlays result
4. Shows prediction label + confidence (e.g., `golden_retriever: 94.2%`) on the live feed
5. Press **`q`** to exit

---

## Key Code

```python
import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2, preprocess_input, decode_predictions
)

def MarvellousImageClassifier():
    # Step 1: Load pre-trained model
    model = MobileNetV2(weights="imagenet")

    # Step 2: Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        # Step 3: Preprocess frame for MobileNetV2
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # BGR → RGB
        img_resized = cv2.resize(img, (224, 224))        # MobileNetV2 expects 224×224
        x = np.expand_dims(img_resized, axis=0).astype(np.float32)
        x = preprocess_input(x)                         # normalize to [-1, 1]

        # Step 4: Predict
        preds = model.predict(x, verbose=0)
        decoded = decode_predictions(preds, top=1)[0][0]
        label = f"{decoded[1]}: {decoded[2]*100:.1f}%"

        # Step 5: Overlay prediction text on frame
        cv2.putText(frame, label, (16, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.imshow("Marvellous Real-time CNN Classification", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

---

## Key Concepts Learned

### MobileNetV2
- A lightweight CNN architecture designed for mobile and real-time applications.
- Pre-trained on **ImageNet** — 1000 classes including animals, objects, vehicles, food, etc.
- Input size: **224×224×3** (RGB, normalized).
- Uses `weights="imagenet"` — no training needed, weights download automatically.

### Preprocessing Pipeline
| Step | Code | Why |
|------|------|-----|
| BGR → RGB | `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` | OpenCV loads as BGR, Keras expects RGB |
| Resize | `cv2.resize(img, (224, 224))` | MobileNetV2 input requirement |
| Add batch dimension | `np.expand_dims(img, axis=0)` | Model expects shape `(1, 224, 224, 3)` |
| Normalize | `preprocess_input(x)` | Scales pixels to `[-1, 1]` range |

### `decode_predictions`
```python
decoded = decode_predictions(preds, top=1)[0][0]
# Returns: (class_id, class_name, probability)
# Example: ('n02099601', 'golden_retriever', 0.942)
```

### Webcam Handling
- `cv2.VideoCapture(0)` — opens default webcam (0 = built-in, 1 = external).
- `cap.read()` — returns `(ret, frame)` where `ret` is a bool (success flag).
- Always call `cap.release()` and `cv2.destroyAllWindows()` to free resources.

---

## How to Run

### Requirements
```bash
pip install tensorflow opencv-python pillow numpy
```

### Run
```bash
python 29_Marvellous_Realtime_Image_Classification.py
```
On first run, MobileNetV2 weights (~14MB) download automatically. Point your webcam at objects and see real-time predictions.

---

## Why This Is a Project

Unlike the other folders (which are step-by-step learning exercises), this file is a **complete, working application**:
- Takes real-world input (webcam)
- Processes it with a production-grade CNN
- Produces meaningful real-time output
- Handles the full pipeline: capture → preprocess → predict → display → cleanup

---

## Libraries Used
- `tensorflow.keras` — MobileNetV2 model, preprocessing, prediction decoding
- `cv2` (OpenCV) — webcam access, frame display, text overlay

---

*OJT Training — Marvellous Infosystems | Deep Learning Track*
