# MNIST Handwritten Digit Classification

This project demonstrates how to classify handwritten digits using a deep learning model built with TensorFlow and the MNIST dataset. The MNIST dataset contains 70,000 grayscale images of handwritten digits (0 to 9) and is a classic benchmark for machine learning models.

## 📋 Project Overview

The goal of this project is to:
- Load and preprocess the MNIST dataset
- Build a simple neural network classifier using TensorFlow and Keras
- Train the model to recognize handwritten digits
- Evaluate model performance on unseen test data

This serves as an educational project to understand the end-to-end process of building a deep learning image classification pipeline.

## 📦 Dataset

- **Dataset:** MNIST Handwritten Digits
- **Source:** [TensorFlow Datasets - MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
- **Size:** 60,000 training images, 10,000 test images
- **Image Size:** 28x28 pixels (grayscale)

## 🛠️ Methodology

### Data Preprocessing
- Normalized pixel values from 0–255 to 0–1
- Batched and cached dataset for efficient training

### Model Architecture
A simple fully connected neural network:
1. Flatten layer to reshape images
2. Dense hidden layer with 128 neurons and ReLU activation
3. Output layer with 10 neurons (one for each digit class)

### Model Training
- Optimizer: Adam
- Loss function: Sparse Categorical Crossentropy (from logits)
- Epochs: 6
- Batch size: 128

### Results
The model typically achieves over **98% accuracy** on the test dataset after training for 6 epochs.

## 📝 Research and Analysis

This project is inspired by the research paper:

> **MNIST Handwritten Digit Classification Based on Convolutional Neural Network with Hyperparameter Optimization**  
> [ResearchGate Link](https://www.researchgate.net/publication/369265604_MNIST_Handwritten_Digit_Classification_Based_on_Convolutional_Neural_Network_with_Hyperparameter_Optimization)

The study showed that tuning hyperparameters significantly improves model performance, achieving near error-free digit recognition.

## 💻 How to Run

1. Install dependencies:
```bash
pip install tensorflow tensorflow-datasets
