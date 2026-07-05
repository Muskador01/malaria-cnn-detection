# Malaria Detection Using Convolutional Neural Networks (CNN)

## Overview

This project presents a Streamlit web application for automated malaria detection using microscopic blood smear cell images. The application uses a trained Convolutional Neural Network (CNN) to classify uploaded blood smear images as either:

- Parasite (Infected)
- Uninfected

## Features

- Upload microscopic blood smear images.
- Automatic image preprocessing.
- CNN-based malaria classification.
- Displays prediction and confidence score.

## Technologies Used

- Python
- TensorFlow/Keras
- Streamlit
- NumPy
- Pillow
- Scikit-learn
- Matplotlib

## Performance
- Test Accuracy: 93.28%
- Training Accuracy (final epoch): 97.33%
- Validation Accuracy (final epoch): 100.00%
- Training Loss (final epoch): 0.1072
- Validation Loss (final epoch): 0.0131
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Epochs: 30

## Project Files

- `app.py` – Streamlit application
- `malaria_cnn_karl.h5` – Trained CNN model
- `malaria_metadata.pkl` – Stores class names, image size and prediction threshold
- `requirements.txt` – Project dependencies

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Karl Njiose

Master's Project in Data Science

Project Title
**Development of a CNN-Based System for Automated Malaria Detection Using Microscopic Blood Smear Cell Images**
