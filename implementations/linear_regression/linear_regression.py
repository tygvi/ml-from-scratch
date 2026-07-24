import numpy as np
class LinearRegression:
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

    def predict(self, X):
        return self.w * X + self.b

    def _compute_cost(self, X, y):
        y_pred = self.w * X + self.b
        cost = (1 / (2 * len(X))) * np.sum((y_pred - y) ** 2)
        return cost

    def _gradient_descent(self, X, y):
        m = len(X)

        for _ in range(self.epochs):
            dj_dw = 0
            dj_db = 0

            for g in range(m):
                prediction = self.predict(X[g])
                error = prediction - y[g]

                dj_dw += error * X[g]
                dj_db += error

            dj_dw = dj_dw / m
            dj_db = dj_db / m

            self.w = self.w - self.learning_rate * dj_dw
            self.b = self.b - self.learning_rate * dj_db