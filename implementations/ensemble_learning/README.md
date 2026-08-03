# Random Forest Classifier from Scratch

A pure NumPy implementation of the **Random Forest** classification algorithm built entirely from first principles.

This implementation combines multiple Decision Trees trained on different bootstrap samples while introducing feature randomness during tree construction. Predictions from all trees are aggregated using majority voting to produce a more robust and generalized classifier.

---

# Overview

Random Forest is an ensemble learning algorithm that reduces the high variance of a single Decision Tree by combining the predictions of many independently trained trees.

Each tree is trained on

- a bootstrap sample of the training dataset
- a random subset of available features at every split

The final prediction is obtained through majority voting.

---

# Features

- Pure NumPy implementation
- Bootstrap sampling
- Random feature selection
- Majority voting
- Recursive CART Decision Trees
- Supports Gini and Entropy
- Configurable number of trees
- Configurable maximum depth
- Multi-dimensional feature support

---

# How Random Forest Works

For every tree

1. Generate a bootstrap sample.
2. Train a Decision Tree.
3. Randomly select a subset of features at every split.
4. Grow the tree independently.

Prediction

1. Every tree predicts independently.
2. Predictions are collected.
3. Majority voting determines the final class.

---

# Bootstrap Sampling

Instead of training every tree on the complete dataset,

Random Forest samples

```text
n samples

with replacement
```

from the original dataset.

This creates diversity among trees.

---

# Random Feature Selection

Unlike a standard Decision Tree,

Random Forest does **not** evaluate every feature at every split.

Instead,

```text
Random subset of features

↓

Best split chosen only from that subset
```

This decorrelates trees and improves generalization.

---

# Majority Voting

Each Decision Tree predicts a class.

Example

```text
Tree 1 → Class 0

Tree 2 → Class 1

Tree 3 → Class 1

Tree 4 → Class 1

Prediction → Class 1
```

---

# Project Structure

```text
ensemble_learning/

├── random_forest.py
└── README.md
```

---

# Class Design

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    criterion="gini",
    max_features=None,
    min_samples_split=2
)
```

---

# Parameters

| Parameter | Description |
|-----------|-------------|
| n_estimators | Number of trees |
| criterion | Gini or Entropy |
| max_depth | Maximum tree depth |
| max_features | Number of randomly selected features |
| min_samples_split | Minimum samples required for splitting |

---

# Methods

## fit()

Builds multiple bootstrap datasets and trains one Decision Tree on each.

---

## predict()

Obtains predictions from every tree and returns the majority vote.

---

## _bootstrap_sample()

Generates a bootstrap sample using random sampling with replacement.

---

## _majority_vote()

Returns the class receiving the largest number of votes.

---

# Time Complexity

Training

```text
O(T × Decision Tree Training)
```

where

```text
T = Number of Trees
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
- No Out-of-Bag score
- No feature importance
- No parallel training
- No probability prediction
- No warm start

---

# Possible Improvements

- Out-of-Bag Error
- Feature Importance
- Parallel Tree Construction
- Predict Probability
- Extra Trees
- Extremely Randomized Trees

---

# Learning Objectives

This implementation demonstrates

- Ensemble Learning
- Bootstrap Aggregation
- Random Feature Sampling
- Variance Reduction
- Majority Voting

---

# References

- Breiman, L. (2001). Random Forests.
- CART
- ISLR
- ESL
