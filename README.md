# Tweet Classification Project

This project focuses on classifying tweets into different categories using various machine learning models. The project covers data preprocessing, feature extraction, model training, hyper-parameter tuning, and performance evaluation.

## Table of Contents

- [Introduction](#introduction)
- [Feature Extraction](#feature-extraction)
- [Model Training](#model-training)
- [Hyper-parameter Tuning](#hyper-parameter-tuning)
- [Performance](#performance)
- [Future Improvements](#future-improvements)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The goal of this project is to build a robust text classification pipeline to categorize tweets into predefined labels. We explore various machine learning algorithms and deep learning models to achieve this task.

## Feature Extraction

Feature extraction is performed using the TfidfVectorizer, which converts text data into numerical features based on term frequency-inverse document frequency (TF-IDF).

## Model Training

The following models are trained and evaluated:

- Logistic Regression
- Support Vector Classifier (SVC)
- Random Forest
- Naive Bayes Classifier
- XGBoost
- TensorFlow Neural Network

Each model is trained on the extracted features, and their performance is evaluated using accuracy and classification reports.

## Hyper-parameter Tuning

GridSearchCV is used to perform hyper-parameter tuning for each model. This process involves testing different combinations of hyper-parameters to find the best settings for each model.

Example for Logistic Regression:
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']
}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_tfidf, y_train)
best_params_lr = grid_search.best_params_
print(f"Best parameters for LogisticRegression: {best_params_lr}")
