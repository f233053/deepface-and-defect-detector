---
tags:
- multi-label-classification
- software-defect-prediction
metrics:
- hamming-loss
- f1
- precision-at-k
---

# Multi-Label Software Defect Prediction Models

This repository contains trained models for predicting multiple software defects simultaneously.

## Models Included

- **SVM**: Support Vector Machine with One-vs-Rest strategy
- **Logistic Regression**: Multi-label logistic regression
- **Perceptron**: Online learning perceptron for multi-label
- **DNN**: Deep Neural Network with sigmoid output layer

## Features

The models use various software metrics including:
- Code complexity metrics
- Size metrics
- Coupling metrics
- Cohesion metrics

## Performance

| Model | Hamming Loss | Micro-F1 | Macro-F1 |
|-------|--------------|----------|----------|
| SVM | 0.15 | 0.78 | 0.72 |
| Logistic Regression | 0.17 | 0.75 | 0.69 |
| Perceptron | 0.20 | 0.70 | 0.64 |
| DNN | 0.13 | 0.82 | 0.76 |

## Usage

```python
import pickle
import numpy as np

# Load model and scaler
with open('defect_svm_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('defect_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Prepare features
features = np.array([...])  # Your feature vector
features_scaled = scaler.transform(features.reshape(1, -1))

# Predict
predictions = model.predict(features_scaled)
probabilities = model.predict_proba(features_scaled)
```

## Training Details

- **Online Learning**: Perceptron uses online learning mode with per-sample weight updates
- **Multi-label Strategy**: One-vs-Rest for SVM and Logistic Regression
- **Hyperparameter Tuning**: Grid search with cross-validation
