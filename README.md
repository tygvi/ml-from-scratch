# ML From Scratch

Machine learning algorithms implemented from scratch using only NumPy, with no scikit-learn training code and no shortcuts.

Most ML libraries hand you a `.fit()` and a `.predict()` and let you move on without ever seeing what happens in between. That's perfectly fine for building applications, but it doesn't always help you understand how the algorithms actually work.

This repository exists because I wanted to understand what gradient descent is really doing to the loss surface, how a decision tree chooses its splits, and why bagging reduces variance instead of bias. The only way I've found to truly internalize these ideas is to implement them myself, watch them break, debug them, and keep going until the results match scikit-learn.

If you're learning machine learning the same way, or you're simply looking for clean, minimal implementations without a framework getting in the way, I hope this repository is useful.

---

## What's Here

- Readable, dependency-light implementations using only NumPy.
- No autograd or hidden abstractions.
- Each algorithm is self-contained and can be understood independently.
- Correctness verified against scikit-learn on standard datasets.
- Docstrings explaining the reasoning behind the implementation, not just the code.
- Tests for every implementation.

---

## Progress

### Implemented

- [x] Linear Regression
- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] Bagging Classifier
- [x] K-Nearest Neighbors

### In Progress / Planned

- [ ] Naive Bayes
- [ ] Support Vector Machine (SVM)
- [ ] K-Means
- [ ] Principal Component Analysis (PCA)

---

## Repository Structure

```text
ml-from-scratch/
├── datasets/                  # Sample datasets used for testing and demos
├── implementations/
│   ├── linear_regression/
│   ├── logistic_regression/
│   ├── decision_tree/
│   ├── ensemble_learning/     # Random Forest and Bagging
│   ├── knn/
│   └── naive_bayes/
├── docs/                      # Notes and derivations
└── tests/
```

Each algorithm directory follows the same layout: the implementation itself, a short README explaining the algorithm, and an example comparing the results against the equivalent scikit-learn implementation.

---

## Built With

- Python 3.9+
- NumPy for numerical computation
- Matplotlib for visualizing decision boundaries and loss curves
- pytest for testing

No ML frameworks or autodiff libraries are used in the implementations. If a gradient needs to be computed, it is derived and implemented by hand.

---

## Installation

```bash
git clone https://github.com/tygvi/ml-from-scratch.git
cd ml-from-scratch
pip install -r requirements.txt
```
```
