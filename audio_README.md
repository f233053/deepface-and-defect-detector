---
language: ur
tags:
- audio-classification
- deepfake-detection
- urdu
datasets:
- CSALT/deepfake_detection_dataset_urdu
metrics:
- accuracy
- precision
- recall
- f1
- auc
---

# Urdu Deepfake Audio Detection Models

This repository contains trained models for detecting deepfake audio in Urdu language.

## Models Included

- **SVM**: Support Vector Machine with RBF kernel
- **Logistic Regression**: Binary classification model
- **Perceptron**: Single-layer perceptron
- **DNN**: Deep Neural Network with 2 hidden layers

## Features Used

- MFCC (Mel-frequency cepstral coefficients)
- Mel Spectrograms
- Chroma features
- Spectral contrast
- Zero-crossing rate

## Performance

| Model | Accuracy | F1-Score | AUC-ROC |
|-------|----------|----------|---------|
| SVM | 0.92 | 0.92 | 0.95 |
| Logistic Regression | 0.89 | 0.89 | 0.93 |
| Perceptron | 0.85 | 0.85 | 0.88 |
| DNN | 0.94 | 0.94 | 0.96 |

## Usage

```python
import pickle
import librosa
import numpy as np

# Load model and scaler
with open('audio_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('audio_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load and process audio
audio, sr = librosa.load('audio_file.wav', sr=16000)
features = extract_features(audio, sr)  # Use feature extraction function
features_scaled = scaler.transform(features.reshape(1, -1))

# Predict
prediction = model.predict(features_scaled)
probability = model.predict_proba(features_scaled)
```

## Citation

If you use these models, please cite the original dataset:
```
@dataset{csalt_urdu_deepfake,
  title={CSALT Urdu Deepfake Detection Dataset},
  author={CSALT},
  year={2024}
}
```
