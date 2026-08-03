# Linear Regression from Scratch

A NumPy implementation of **Simple Linear Regression** trained using **Batch Gradient Descent**.

This project implements linear regression from first principles without relying on machine learning libraries such as scikit-learn. The objective is to understand how linear models are optimized through gradient descent by manually deriving and implementing the learning algorithm.

---

## Overview

Linear Regression is one of the most fundamental supervised learning algorithms used for predicting continuous numerical values.

Given an input feature \(x\), the model learns a linear relationship

```text
ŷ = wx + b
```

where

- **w** is the weight (slope)
- **b** is the bias (intercept)

The parameters are optimized by minimizing the Mean Squared Error (MSE) using Batch Gradient Descent.

---

## Features

- Pure NumPy implementation
- No scikit-learn dependency
- Batch Gradient Descent optimization
- Object-oriented design
- Configurable learning rate
- Configurable number of training epochs
- Separate training and prediction methods
- Mean Squared Error cost function

---

## Mathematical Background

### Hypothesis

\[
\hat{y}=wx+b
\]

---

### Cost Function

The model minimizes the Mean Squared Error (MSE)

```text
         m
J(w,b) = ─── Σ (ŷᵢ − yᵢ)²
        2m i=1
```

where

- \(m\) = number of training examples
- \(\hat{y}\) = predicted value
- \(y\) = actual value

---

### Gradient Computation

Gradient with respect to the weight

```
∂J      1
── = ───── Σ (ŷᵢ − yᵢ)xᵢ
∂w      m
```

Gradient with respect to the bias

```
∂J      1
── = ───── Σ (ŷᵢ − yᵢ)
∂b      m
```

---

### Parameter Update

The parameters are updated after every epoch using

w ← w − α ∂J/∂w

b ← b − α ∂J/∂b

where

- \(\alpha\) is the learning rate.

---

# Project Structure

```text
linear_regression/
│
├── linear_regression.py
└── README.md
```

---

# Class Design

```python
LinearRegression(
    learning_rate=0.01,
    epochs=1000
)
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| learning_rate | Step size used during gradient descent |
| epochs | Number of optimization iterations |

---

# Methods

## fit(X, y)

Trains the model by performing Batch Gradient Descent.

Returns the fitted model.

---

## predict(X)

Generates predictions using

\[
\hat y = wx+b
\]

---

## _compute_cost(X, y)

Computes the Mean Squared Error cost.

This method is useful for monitoring convergence during training.

---

## _gradient_descent(X, y)

Internal optimization routine responsible for

- computing gradients
- updating parameters
- minimizing the cost function

---

# Training Workflow

The training procedure follows these steps:

1. Initialize weight and bias to zero.
2. Perform a forward pass to compute predictions.
3. Compute prediction errors.
4. Calculate gradients for both parameters.
5. Update weight and bias using gradient descent.
6. Repeat for the specified number of epochs.

---

# Usage

```python
import numpy as np
from linear_regression import LinearRegression

X = np.array([1,2,3,4,5])
y = np.array([3,5,7,9,11])

model = LinearRegression(
    learning_rate=0.01,
    epochs=1000
)

model.fit(X,y)

predictions = model.predict(X)

print(predictions)
```

---

# Time Complexity

## Training

For

- \(m\) training examples
- \(e\) epochs

the training complexity is

\[
O(em)
\]

---

## Prediction

\[
O(n)
\]

where \(n\) is the number of samples.

---

# Space Complexity

Training

\[
O(1)
\]

excluding the input dataset.

---

# Current Limitations

This implementation intentionally focuses on the core learning algorithm.

Current limitations include:

- Supports only **single-feature** linear regression
- Uses Batch Gradient Descent only
- No feature normalization
- No regularization (L1/L2)
- No early stopping
- No convergence tolerance
- No learning-rate scheduling
- No training history visualization

---

# Possible Improvements

Future versions may include

- Multiple Linear Regression
- Vectorized gradient computation
- Mini-batch Gradient Descent
- Stochastic Gradient Descent (SGD)
- L1/L2 Regularization
- Polynomial Regression
- Learning rate scheduling
- Early stopping
- Training loss history
- R² score implementation

---

# Learning Objectives

This implementation was created to gain a deeper understanding of

- Linear Regression
- Mean Squared Error
- Gradient Descent
- Cost minimization
- Numerical optimization
- Object-oriented implementation of machine learning algorithms

---

# References

- Andrew Ng — Machine Learning Specialization
- The Elements of Statistical Learning
- An Introduction to Statistical Learning (ISLR)
- Pattern Recognition and Machine Learning — Christopher Bishop
