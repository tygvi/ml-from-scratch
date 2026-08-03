# Gaussian Naive Bayes from Scratch

A pure NumPy implementation of the **Gaussian Naive Bayes** classifier built entirely from first principles.

This implementation models every feature as a Gaussian (Normal) distribution for each class and performs classification using **Bayes' Theorem**. To improve numerical stability, all probability computations are carried out in **log-space**, preventing floating-point underflow when multiplying many small probabilities.

The implementation is benchmarked against **scikit-learn's GaussianNB** on the Breast Cancer dataset.

---

# Overview

Gaussian Naive Bayes is a probabilistic supervised learning algorithm based on Bayes' Theorem.

The model assumes

- Features are conditionally independent given the class.
- Every feature follows a Gaussian (Normal) distribution.

For each class, the algorithm learns

- Prior probability
- Mean of every feature
- Variance of every feature

During prediction, the posterior probability is computed for every class and the class with the highest probability is returned.

---

# Features

- Pure NumPy implementation
- Gaussian likelihood estimation
- Automatic prior probability estimation
- Log-probability computation for numerical stability
- Multi-feature support
- Multi-class classification support
- Benchmark comparison with scikit-learn
- Object-oriented implementation

---

# Mathematical Background

## Bayes' Theorem

The classifier predicts

```text
              P(X | C) P(C)
P(C | X) = ------------------
                 P(X)
```

Since P(X) is identical for every class, classification becomes

```text
Prediction = argmax P(X|C) P(C)
```

---

## Naive Independence Assumption

The algorithm assumes that all features are conditionally independent.

```text
P(X|C)

= P(x₁|C)

× P(x₂|C)

× ...

× P(xₙ|C)
```

Although this assumption is often unrealistic, Naive Bayes performs remarkably well on many real-world datasets.

---

## Gaussian Probability Density Function

Each feature is modeled using a Normal distribution.

```text
                     -(x-μ)²
                   ------------
                      2σ²
               e
P(x|C) = ---------------------------
           √(2πσ²)
```

where

- μ = feature mean
- σ² = feature variance

---

## Log Probability

Instead of multiplying many tiny probabilities,

```text
P₁ × P₂ × P₃ × ...
```

the implementation computes

```text
log(P₁)

+ log(P₂)

+ log(P₃)

+ ...
```

This significantly improves numerical stability.

---

# How the Algorithm Works

Training consists of

1. Computing class prior probabilities.
2. Computing the mean of every feature for every class.
3. Computing the variance of every feature for every class.

Prediction consists of

1. Computing the log prior.
2. Computing the log Gaussian likelihood for every feature.
3. Summing all log probabilities.
4. Selecting the class with the highest score.

---

# Project Structure

```text
gaussian_naive_bayes/
│
├── gaussian_naive_bayes.py
└── README.md
```

---

# Class Design

```python
GaussianNaiveBayes()
```

---

# Methods

## fit(X, y)

Learns

- class priors
- feature means
- feature variances

Returns the fitted model.

---

## predict(X)

Predicts class labels for every sample.

---

## _predict_single(x)

Computes the posterior score for one sample and returns the most probable class.

---

## compute_priors()

Computes

```text
P(Class)
```

for every class.

---

## compute_class_stats()

Computes

- mean
- variance

for every feature inside every class.

---

## gaussian_pdf()

Computes the Gaussian probability density.

---

## log_gaussian_pdf()

Computes the logarithm of the Gaussian probability density.

This method is used during prediction to improve numerical stability.

---

# Usage

```python
from gaussian_naive_bayes import GaussianNaiveBayes

model = GaussianNaiveBayes()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

---

# Benchmark

The implementation was evaluated on the Breast Cancer dataset.

| Model | Accuracy |
|---------|---------:|
| From Scratch GaussianNB | **92.98%** |
| scikit-learn GaussianNB | **93.86%** |

The custom implementation achieves performance very close to scikit-learn while relying only on NumPy.

---

# Time Complexity

Let

```text
n = number of samples

d = number of features

k = number of classes
```

### Training

```text
O(nd)
```

---

### Prediction

```text
O(kd)
```

per sample.

---

# Space Complexity

The classifier stores

- priors
- feature means
- feature variances

Space complexity

```text
O(kd)
```

---

# Current Limitations

This implementation focuses on understanding Gaussian Naive Bayes.

Current limitations include

- Gaussian features only
- No categorical Naive Bayes
- No Multinomial Naive Bayes
- No Bernoulli Naive Bayes
- Equal feature importance
- No incremental learning
- No probability calibration

---

# Possible Improvements

Future versions may include

- Bernoulli Naive Bayes
- Multinomial Naive Bayes
- Categorical Naive Bayes
- Partial Fit
- Probability calibration
- Vectorized likelihood computation
- Feature importance visualization

---

# Learning Objectives

This implementation was created to understand

- Bayes' Theorem
- Probabilistic classification
- Gaussian distributions
- Log probabilities
- Numerical stability
- Maximum A Posteriori (MAP) estimation
- Statistical machine learning

---

# References

- Tom M. Mitchell — Machine Learning
- Pattern Recognition and Machine Learning — Christopher Bishop
- An Introduction to Statistical Learning (ISLR)
- The Elements of Statistical Learning
- scikit-learn Documentation
