# K-Nearest Neighbors (KNN) from Scratch

A NumPy implementation of the **K-Nearest Neighbors (KNN)** classification algorithm.

This project implements KNN from first principles without relying on machine learning libraries for the prediction algorithm. The implementation uses Euclidean distance to identify the nearest neighbors and majority voting to classify unseen samples.

For benchmarking purposes, predictions are compared against scikit-learn's `KNeighborsClassifier`.

---

# Overview

K-Nearest Neighbors is a **non-parametric**, **instance-based** supervised learning algorithm.

Unlike many machine learning models, KNN does not learn explicit parameters during training. Instead, it stores the training dataset and performs classification only when predictions are requested.

The prediction process consists of four steps:

1. Compute the distance from the query point to every training sample.
2. Select the **K nearest neighbors**.
3. Count the class labels among those neighbors.
4. Assign the majority class to the query point.

---

# Features

- Pure NumPy implementation
- Euclidean distance metric
- Majority voting classification
- Configurable number of neighbors (K)
- Object-oriented implementation
- Benchmark comparison with scikit-learn
- Supports multi-dimensional feature vectors

---

# Mathematical Background

## Euclidean Distance

The distance between two samples is computed using

```text
                _______________________
               /
d(x,y) = √ Σ (xᵢ − yᵢ)²
```

The nearest neighbors are the samples with the smallest Euclidean distances.

---

## Majority Voting

After selecting the K nearest neighbors,

```text
Prediction = Most Frequent Label
```

Example

```text
Neighbors

1
1
0

Prediction → 1
```

---

# Why Feature Scaling?

Since KNN is a distance-based algorithm, features with larger numerical ranges can dominate the distance calculation.

For this reason, the dataset is standardized before training using

```text
StandardScaler
```

This preprocessing step is not part of the KNN algorithm itself but ensures fair distance computations.

---

# Project Structure

```text
knn/
│
├── knn.py
└── README.md
```

---

# Class Design

```python
Knn(k=3)
```

---

# Parameters

| Parameter | Description |
|-----------|-------------|
| k | Number of nearest neighbors used for prediction |

---

# Methods

## fit(X_train, y_train)

Stores the training dataset.

Unlike most machine learning algorithms, no optimization or parameter learning occurs during training.

---

## predict(X_test)

Predicts labels for unseen samples by

- computing distances
- selecting K nearest neighbors
- performing majority voting

Returns

```text
NumPy array of predicted labels
```

---

## calculate_distance(point_A, point_B)

Computes Euclidean distance using

```python
np.linalg.norm(point_A - point_B)
```

---

## majority_count(neighbors)

Determines the predicted class by counting neighbor labels using Python's `Counter` class.

---

# Training Workflow

1. Store the training dataset.
2. Receive a query sample.
3. Compute distances to all training samples.
4. Sort distances.
5. Select the K closest samples.
6. Perform majority voting.
7. Return the predicted class.

---

# Usage

```python
from knn import Knn

model = Knn(k=3)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Benchmark

The implementation is validated by comparing its predictions against

```python
sklearn.neighbors.KNeighborsClassifier
```

using the same dataset and preprocessing pipeline.

Prediction accuracy is evaluated using

```python
accuracy_score()
```

---

# Time Complexity

## Training

Since training only stores the dataset,

```text
O(1)
```

---

## Prediction

For

- n training samples
- d features

each prediction requires

```text
O(nd)
```

Sorting the distances contributes

```text
O(n log n)
```

Overall prediction complexity

```text
O(n log n)
```

---

# Space Complexity

The algorithm stores the complete training dataset.

```text
O(nd)
```

---

# Current Limitations

This implementation focuses on the core KNN algorithm.

Current limitations include

- Brute-force nearest neighbor search
- Euclidean distance only
- Classification only
- Uniform voting
- Full sorting of distances
- No KD-Tree or Ball Tree optimization
- No weighted neighbors

---

# Possible Improvements

Future versions may include

- KD-Tree search
- Ball Tree search
- Weighted KNN
- Manhattan distance
- Minkowski distance
- Cosine similarity
- KNN Regression
- Distance-weighted voting
- Partial sorting using `heapq` for improved efficiency

---

# Learning Objectives

This implementation was created to understand

- Instance-based learning
- Distance metrics
- Euclidean geometry
- Lazy learning algorithms
- Majority voting
- Nearest neighbor search
- Classification without explicit training

---

# References

- Cover, T. & Hart, P. (1967). Nearest Neighbor Pattern Classification.
- An Introduction to Statistical Learning (ISLR)
- Pattern Recognition and Machine Learning — Christopher Bishop
- scikit-learn Documentation
