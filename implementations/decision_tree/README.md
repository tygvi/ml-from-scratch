# Decision Tree Classifier from Scratch

A pure NumPy implementation of a **Decision Tree Classifier** built entirely from scratch.

This project implements a recursive binary decision tree capable of handling multi-dimensional datasets using either **Gini Impurity** or **Entropy (Information Gain)** as the splitting criterion.

The implementation does not rely on any machine learning libraries for tree construction or prediction and is designed to demonstrate the internal mechanics of Decision Trees.

---

# Overview

Decision Trees are supervised learning algorithms that recursively partition the feature space into smaller and more homogeneous regions.

At each node, the algorithm searches for the feature and threshold that maximize the reduction in impurity.

The process continues recursively until a stopping criterion is met.

---

# Features

- Pure NumPy implementation
- Recursive tree construction
- Supports multiple input features
- Binary splits
- Configurable maximum tree depth
- Configurable minimum samples required for splitting
- Supports both
  - Gini Impurity
  - Entropy (Information Gain)
- Automatic leaf node generation
- Recursive prediction

---

# Decision Tree Structure

Each node stores

```text
Feature Index
Threshold
Left Child
Right Child
Leaf Value (if leaf)
```

Internal nodes contain

```text
Feature
Threshold
Left Child
Right Child
```

Leaf nodes contain

```text
Predicted Class
```

---

# Mathematical Background

## Entropy

Entropy measures the uncertainty within a node.

```text
Entropy = - Σ pᵢ log₂(pᵢ)
```

Lower entropy indicates purer nodes.

---

## Gini Impurity

Gini measures the probability of incorrectly classifying a randomly selected sample.

```text
Gini = Σ pᵢ (1 − pᵢ)
```

Lower Gini values indicate better class separation.

---

## Information Gain

The quality of a split is measured using

```text
Information Gain

= Parent Impurity
− Weighted Child Impurity
```

The split with the highest information gain is selected.

---

# How the Algorithm Works

For every node

1. Iterate over every feature.
2. Extract all unique feature values.
3. Compute candidate thresholds using adjacent midpoints.
4. Split the dataset.
5. Compute impurity reduction.
6. Select the best split.
7. Recursively build left and right subtrees.
8. Continue until a stopping condition is reached.

---

# Stopping Criteria

Tree growth stops when any of the following conditions are satisfied.

- Maximum depth reached
- Number of samples is smaller than `min_samples_split`
- All samples belong to the same class
- No split produces positive information gain

In each case, a leaf node is created using the majority class.

---

# Project Structure

```text
decision_tree/
│
├── decision_tree.py
└── README.md
```

---

# Class Design

```python
DecisionTreeClassifier(
    criterion="gini",
    max_depth=None,
    min_samples_split=2
)
```

---

# Parameters

| Parameter | Description |
|------------|-------------|
| criterion | Splitting criterion ("gini" or "entropy") |
| max_depth | Maximum allowed tree depth |
| min_samples_split | Minimum samples required before splitting |

---

# Methods

## fit(X, y)

Builds the complete decision tree recursively.

Returns the fitted model.

---

## predict(X)

Traverses the learned tree to classify each sample.

Returns a NumPy array of predictions.

---

## entropy(y)

Computes the entropy of a node.

---

## gini(y)

Computes the Gini impurity.

---

## information_gain(...)

Computes the impurity reduction produced by a candidate split.

---

## best_split(...)

Searches every feature and every candidate threshold to identify the split that maximizes Information Gain.

Returns

- best feature
- best threshold
- best gain

---

## build_tree(...)

Recursively constructs the Decision Tree.

Each recursive call creates either

- an internal decision node
- or a leaf node

depending on the stopping criteria.

---

## _predict(...)

Recursively traverses the tree until a leaf node is reached.

The class stored inside the leaf node is returned as the prediction.

---

# Usage

```python
from decision_tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_split=2
)

tree.fit(X_train, y_train)

predictions = tree.predict(X_test)
```

---

# Time Complexity

Let

```text
n = number of samples
d = number of features
```

### Training

Because every feature and multiple thresholds are evaluated recursively,

the worst-case complexity is approximately

```text
O(d · n² log n)
```

although it depends on tree balance and the number of unique feature values.

---

### Prediction

Prediction follows one path from the root to a leaf.

Average case

```text
O(log n)
```

Worst case

```text
O(n)
```

---

# Space Complexity

Tree storage

```text
O(n)
```

Recursive stack

```text
O(tree depth)
```

---

# Current Limitations

This implementation focuses on understanding the Decision Tree learning process.

Current limitations include

- Binary splits only
- Numerical features only
- Brute-force threshold search
- No pruning
- No handling of missing values
- No feature importance calculation
- No support for sample weights
- No random feature selection (Random Forest)

---

# Possible Improvements

Future versions may include

- Cost Complexity Pruning
- Pre-pruning strategies
- Feature Importance
- Random feature sampling
- Categorical feature support
- Missing value handling
- Vectorized split search
- Faster threshold computation
- Parallel split evaluation

---

# Learning Objectives

This implementation was created to gain a deeper understanding of

- Recursive tree construction
- Binary partitioning
- Information Gain
- Entropy
- Gini Impurity
- Divide-and-Conquer algorithms
- Recursive prediction
- Tree-based machine learning algorithms

---

# References

- Breiman, Friedman, Olshen & Stone — Classification and Regression Trees (CART)
- The Elements of Statistical Learning
- An Introduction to Statistical Learning (ISLR)
- Pattern Recognition and Machine Learning — Christopher Bishop
- Andrew Ng — Machine Learning Specialization
