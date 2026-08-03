# Logistic Regression from Scratch

A NumPy implementation of **Binary Logistic Regression** trained using **Batch Gradient Descent**.

This project implements logistic regression from first principles without relying on machine learning libraries such as scikit-learn. The model learns to classify data into two classes by estimating the probability of an input belonging to the positive class.

---

# Overview

Unlike Linear Regression, which predicts continuous values, Logistic Regression predicts the probability that a sample belongs to a particular class.

The model computes

```text
z = wx + b
```

and passes it through the Sigmoid function

```text
          1
σ(z) = ---------
       1 + e⁻ᶻ
```

to produce probabilities between 0 and 1.

Predictions are obtained using a threshold of **0.5**.

```text
Probability ≥ 0.5 → Class 1

Probability < 0.5 → Class 0
```

---

# Features

- Pure NumPy implementation
- Binary classification
- Batch Gradient Descent optimization
- Binary Cross-Entropy (Log Loss)
- Sigmoid activation
- Probability prediction
- Class prediction
- Object-oriented implementation

---

# Mathematical Background

## Linear Combination

```text
z = wx + b
```

---

## Sigmoid Function

```text
          1
σ(z) = ---------
       1 + e⁻ᶻ
```

The sigmoid function maps any real-valued input to the interval

```text
(0,1)
```

making it suitable for probability estimation.

---

## Cost Function

Binary Cross Entropy Loss

```text
J(w,b) = -(1/m) Σ
          [y log(ŷ) + (1-y) log(1-ŷ)]
```

where

- m = number of training examples
- y = true label
- ŷ = predicted probability

To prevent numerical instability, predicted probabilities are clipped before computing the logarithm.

---

## Gradient Computation

Weight gradient

```text
∂J      1
── = ───── Σ (ŷᵢ − yᵢ)xᵢ
∂w      m
```

Bias gradient

```text
∂J      1
── = ───── Σ (ŷᵢ − yᵢ)
∂b      m
```

---

## Parameter Update

```text
w ← w − α(∂J/∂w)

b ← b − α(∂J/∂b)
```

where α is the learning rate.

---

# Project Structure

```text
logistic_regression/
│
├── logistic_regression.py
└── README.md
```

---

# Class Design

```python
LogisticRegression(
    learning_rate=0.01,
    epochs=1000
)
```

---

# Parameters

| Parameter | Description |
|-----------|-------------|
| learning_rate | Step size used during gradient descent |
| epochs | Number of optimization iterations |

---

# Methods

## fit(X, y)

Trains the model using Batch Gradient Descent.

Returns the fitted model.

---

## predict_proba(X)

Computes the probability of each sample belonging to the positive class.

Returns values between

```text
0 and 1
```

---

## predict(X)

Converts predicted probabilities into binary class labels.

Classification rule

```text
Probability ≥ 0.5 → 1

Probability < 0.5 → 0
```

---

## _sigmoid(z)

Applies the sigmoid activation function.

---

## _compute_cost(X, y)

Computes Binary Cross Entropy Loss.

Predicted probabilities are clipped to prevent

```text
log(0)
```

errors.

---

## _gradient_descent(X, y)

Internal optimization routine responsible for

- computing gradients
- updating parameters
- minimizing the loss

---

# Training Workflow

The training process follows these steps

1. Initialize weight and bias to zero.
2. Compute the linear combination.
3. Apply the sigmoid function.
4. Compute Binary Cross Entropy Loss.
5. Calculate gradients.
6. Update parameters.
7. Repeat for the specified number of epochs.

---

# Usage

```python
import numpy as np
from logistic_regression import LogisticRegression

X = np.array([0.2, 0.8, 1.3, 2.1, 3.0])
y = np.array([0, 0, 0, 1, 1])

model = LogisticRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X, y)

probabilities = model.predict_proba(X)

predictions = model.predict(X)
```

---

# Time Complexity

## Training

For

- m training examples
- e epochs

Training complexity

```text
O(em)
```

---

## Prediction

```text
O(n)
```

where n is the number of samples.

---

# Space Complexity

```text
O(1)
```

excluding the input dataset.

---

# Current Limitations

This implementation focuses on understanding the core learning algorithm.

Current limitations include

- Supports only binary classification
- Supports only a single input feature
- Uses Batch Gradient Descent only
- No regularization (L1/L2)
- No early stopping
- No learning-rate scheduling
- No training history visualization
- No multi-class classification

---

# Possible Improvements

Future versions may include

- Multiple feature support
- Vectorized gradient computation
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- L1 and L2 Regularization
- Softmax Regression
- One-vs-Rest classification
- Learning rate scheduling
- Early stopping
- Training loss visualization

---

# Learning Objectives

This implementation was created to gain a deeper understanding of

- Binary classification
- Logistic Regression
- Sigmoid activation
- Binary Cross Entropy Loss
- Gradient Descent
- Probability estimation
- Object-oriented implementation of machine learning algorithms

---

# References

- Andrew Ng — Machine Learning Specialization
- An Introduction to Statistical Learning (ISLR)
- The Elements of Statistical Learning
- Pattern Recognition and Machine Learning — Christopher Bishop
