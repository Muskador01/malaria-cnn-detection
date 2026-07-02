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

## Performamce
- Training Accuracy: 97.33%
- Validation Accuracy: 95.12%
- Training Loss: 0.1099
- Validation Loss: 0.1551
- Optimizer: Adam
- Loss Function: Binary Cross-Entropy
- Epochs: 30

## Project Files

- `app.py` – Streamlit application
- `malaria_cnn_karl.h5` – Trained CNN model
- `malaria_metadata.pkl` – Stores class names, image suze and prediction threshold
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