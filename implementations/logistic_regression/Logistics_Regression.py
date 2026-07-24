import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.w = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self._gradient_descent(X, y)
        return self

    def predict_proba(self, X):
        z = self.w * X + self.b
        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _compute_cost(self, X, y):
        y_pred = self.predict_proba(X)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cost = (-1 / len(X)) * np.sum(
            y * np.log(y_pred) +
            (1 - y) * np.log(1 - y_pred))
        return cost

    def _gradient_descent(self, X, y):
        m = len(X)

        for _ in range(self.epochs):
            dj_dw = 0
            dj_db = 0

            for i in range(m):
                prediction = self.predict_proba(X[i])
                error = prediction - y[i]

                dj_dw += error * X[i]
                dj_db += error

            dj_dw = dj_dw / m
            dj_db = dj_db / m

            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db