# Data Science Projects: CIFAR-10 and Dementia Image Classification

This repository contains two machine learning projects focusing on image classification using the Vision Transformer (ViT) model. Both projects utilize PyTorch Lightning for model training and evaluation.

## Project 1: CIFAR-10 Image Classification

### Overview
- The project aims to classify images from the CIFAR-10 dataset using a ViT model.
- It includes data loading, preprocessing, defining a ViT model, training, and evaluation.

### Key Components
- **Data Loading**: A subset of the CIFAR-10 dataset is loaded for training and testing.
- **Preprocessing**: Images are preprocessed using `torchvision.transforms` and the ViT image processor.
- **Model Definition**: A LightningModule for the ViT model is defined with custom label mappings.
- **Training**: The model is trained using PyTorch Lightning's Trainer class with an early stopping callback.
- **Evaluation**: Model performance is evaluated on a test set and visualized using a confusion matrix.

## Project 2: Dementia Image Classification

### Overview
- This project focuses on classifying images from a dementia dataset into different categories of dementia severity.
- It follows a similar approach to Project 1, tailored to the dementia dataset.

### Key Components
- **Data Loading**: A subset of the dementia dataset is used for training and testing.
- **Preprocessing**: Custom transformations are applied to the images for model compatibility.
- **Model Definition**: A ViT model is adapted to classify four categories of dementia severity.
- **Training**: Training is carried out with PyTorch Lightning and early stopping.
- **Evaluation**: The model's accuracy and loss are assessed, and predictions are visualized.

## Technologies Used
- Python
- PyTorch
- PyTorch Lightning
- Transformers
- Torchvision
- Numpy
- Sklearn

---

[← Portfolio index](../../README.md)
