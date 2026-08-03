# Bagging Classifier from Scratch

A pure NumPy implementation of the **Bootstrap Aggregating (Bagging)** algorithm.

This implementation trains multiple Decision Trees on independently generated bootstrap datasets and combines their predictions using majority voting.

Unlike Random Forest, every Decision Tree is allowed to evaluate **all available features** when selecting the best split.

---

# Overview

Bagging (Bootstrap Aggregating) is one of the earliest ensemble learning techniques.

Its objective is to reduce the variance of unstable models such as Decision Trees.

The algorithm

- generates multiple bootstrap datasets
- trains one model on each dataset
- combines predictions using majority voting

---

# Features

- Pure NumPy implementation
- Bootstrap sampling
- Majority voting
- Multiple Decision Trees
- Configurable number of estimators
- Supports Gini and Entropy
- Recursive CART Decision Trees

---

# How Bagging Works

For every estimator

1. Generate a bootstrap sample.
2. Train a Decision Tree.
3. Repeat for all estimators.

Prediction

1. Every tree predicts independently.
2. Majority voting produces the final prediction.

---

# Difference Between Bagging and Random Forest

| Bagging | Random Forest |
|----------|---------------|
| Bootstrap Sampling | Bootstrap Sampling |
| Uses all features | Random subset of features |
| Trees are more correlated | Trees are less correlated |
| Higher variance | Lower variance |

---

# Bootstrap Sampling

Each tree receives a randomly generated dataset sampled

```text
with replacement
```

This introduces diversity into the ensemble.

---

# Majority Voting

Example

```text
Tree 1 → 0

Tree 2 → 1

Tree 3 → 1

Prediction → 1
```

---

# Project Structure

```text
ensemble_learning/

├── bagging_classifier.py
└── README.md
```

---

# Class Design

```python
BaggingClassifier(
    n_estimators=100,
    max_depth=None,
    criterion="gini",
    min_samples_split=2
)
```

---

# Methods

- fit()
- predict()
- _bootstrap_sample()
- _majority_vote()

---

# Time Complexity

Training

```text
O(T × Decision Tree Training)
```

Prediction

```text
O(T × Tree Prediction)
```

---

# Space Complexity

```text
O(T × Tree Size)
```

---

# Current Limitations

- Classification only
- No weighted voting
- No probability prediction
- No parallel execution
- No Out-of-Bag evaluation

---

# Possible Improvements

- Parallel Bagging
- Probability Prediction
- Weighted Voting
- Out-of-Bag Error
- Generic Base Estimators

---

# Learning Objectives

This implementation demonstrates

- Bootstrap Sampling
- Ensemble Learning
- Variance Reduction
- Majority Voting

---

# References

- Leo Breiman — Bagging Predictors (1996)
- ISLR
- ESL
