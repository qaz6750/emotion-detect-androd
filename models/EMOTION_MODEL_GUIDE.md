# Emotion Recognition ONNX Model Guide

## Model Information
- **Input**: Grayscale face image (48×48 pixels)
- **Output**: 7 emotion probabilities
- **Format**: ONNX (compatible with Android ONNX Runtime)

## Emotion Classes
The model predicts the following 7 emotions:
0. Angry (愤怒)
1. Disgust (厌恶) 
2. Fear (恐惧)
3. Happy (快乐)
4. Sad (悲伤)
5. Surprise (惊讶)
6. Neutral (中性)

## Input Preprocessing
1. **Face Detection**: Extract face region from image
2. **Resize**: Resize face to 48×48 pixels
3. **Grayscale**: Convert to grayscale (1 channel)
4. **Normalize**: Normalize pixel values to [0, 1] range
5. **Tensor Format**: NCHW format [1, 1, 48, 48]

## Usage Example (Python)
```python
import cv2
import numpy as np
import onnxruntime as ort

# Load model
session = ort.InferenceSession('fer2013_resnet_emotion.onnx')

# Preprocess face image
face_gray = cv2.resize(face_image, (48, 48))
face_gray = face_gray.astype(np.float32) / 255.0
input_tensor = face_gray.reshape(1, 1, 48, 48)

# Run inference
outputs = session.run(None, {'input': input_tensor})
emotions = outputs[0][0]

# Get predicted emotion
emotion_idx = np.argmax(emotions)
confidence = emotions[emotion_idx]
```

## Android Implementation Notes
- Use ONNX Runtime Android
- Input tensor: FloatArray of size 2304 (48×48)
- Output tensor: FloatArray of size 7
- Apply softmax to get probabilities
- Use argmax to get predicted class
