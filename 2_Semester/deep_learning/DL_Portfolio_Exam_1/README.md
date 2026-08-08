# Data Science Coursework Project: Perceptron and SMOTE Analysis

## Project Overview
This project, part of a data science coursework, focuses on implementing and analyzing a Perceptron model and Synthetic Minority Over-sampling Technique (SMOTE) in Python. The project is divided into several exercises, each addressing different aspects of machine learning, from basic Perceptron operations to handling imbalanced datasets using over-sampling techniques.

## Contents and Methods

### Exercise 1: Perceptron Implementation
- Implementation of a basic Perceptron model.
- The model is tested on a given tensor with specific weights and bias.
- The output is a 1-dimensional tensor without an activation function.

### Exercise 2: Tensor Modeling
- Designing a tensor to model tagging data for a hypothetical scenario.
- The tensor includes dimensions for user identifiers, product identifiers, and tags.
- Analysis of the tensor's size and the implications of changes in the data (e.g., new tags).

### Exercise 3: SMOTE Analysis
- Discussion on the usefulness of SMOTE in imbalanced datasets.
- Examples include fraud detection, oil spill detection in satellite images, and cancerous pixel identification in mammograms.
- Comparison of SMOTE with traditional oversampling methods.

### Exercise 4: Perceptron and Data Sampling
- Implementation of a Perceptron class in Python.
- Loading and preprocessing of a dataset (`portfolio_data_wise_2022.csv`).
- Analysis of class distribution in the dataset.
- Visualization of data points using a scatter plot.
- Comparison of Perceptron performance with different data sampling techniques: plain data, SMOTE, and random oversampling.
- Evaluation of results using metrics like accuracy, balanced accuracy, and confusion matrices.

### Exercise 5: Expectations vs. Results
- Comparison of expected outcomes with actual results from the Perceptron model trained on different data sampling methods.
- Visualization of decision boundaries for original and SMOTE-resampled data.
- Discussion on the impact of SMOTE and potential adjustments for improved results.

## Technologies Used
- Python
- Libraries: Pandas, NumPy, Matplotlib, Plotly Express, scikit-learn, imbalanced-learn

---

[← Portfolio index](../../../README.md)
