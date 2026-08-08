# Data Science Coursework Project: Neural Network-Based Credit Default Prediction

## Project Overview
This project, a part of data science coursework, focuses on developing a neural network model to predict credit default. It extends our previous work with a bank, analyzing anonymized data from 1000 customers using machine learning and now neural networks. The aim is to improve the prediction accuracy for credit defaults.

## Contents and Methods

### Data Ingestion and Preprocessing
- Downloading and preprocessing data from the UCI Machine Learning Repository's South German Credit dataset.
- Initial exploratory data analysis and data preparation for visualization and machine learning.

### Exploratory Data Analysis
- Analysis of correlations between various features and credit default.
- Visualization of key variables like credit compliance, account status, credit amount, and debtor's age.

### Neural Network Modeling
- Implementation of a flexible multilayer perceptron model in PyTorch.
- Experimentation with various architectures to determine the most effective model structure.

### Class Imbalance Handling
- Analysis of class distribution in the dataset.
- Implementation of strategies like class weight adjustment to handle class imbalance.

### Hyperparameter Tuning and Model Evaluation
- Hyperparameter tuning using `ray.tune`.
- Evaluation of the model's performance on the test set.
- Comparison of neural network performance with previous machine learning models.

### Results and findings

Best balanced validation accuracy **0.766**, test accuracy **69.96%**.

The network beat the previous best model on this data — an SVM — but the gain in
balanced accuracy was **marginal**, and does not obviously justify the added
complexity on a dataset of 1,000 rows. That is the honest conclusion of the
project: a neural network was applicable here, not clearly warranted.

Class weighting mattered more than architecture. With 1,000 customers and a
skewed default rate, plain accuracy is a misleading target — a model that never
predicts default scores well on it — which is why balanced accuracy is the
figure reported above.

## Technologies Used
- Python
- Libraries: Pandas, NumPy, Matplotlib, Plotly Express, PyTorch, ray.tune

---

[← Portfolio index](../../../README.md)
