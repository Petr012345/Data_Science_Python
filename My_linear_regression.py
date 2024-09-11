import numpy as np

# My implementation of Linear regression model

class Linear_model:

    # Initialization, l - number of feathures
    def __init__(self, l):
        self.W = np.zeros(l)
        # self.W = None

    # Fitting function, repeat while accuracy(or another metric) becomes better
    # X - train dataframe with feathures
    # Y - train dataframe with targets
    # alf - coefficient of education speed
    def gradient_descent(self, X, Y, alf):
        w_delta = (2/X.shape[1])*(X.T.dot(X.dot(self.W.T)-Y))
        self.W -= alf*w_delta

    # One of quality evaluating functions
    # Y - true results
    # P_Y - predicted results
    def loss_func(self, Y, P_Y):
        return np.sum(np.square(Y-P_Y))

    # Ideal mathematical solution (unstable)
    # X - train dataframe with feathures
    # Y - train dataframe with targets
    # lam - regularization variable (higher value -> less accuracy, lower value -> less stability)
    def formulated_fit(self, X, Y, lam=0.0):
        reg_matrix = lam * np.eye(X.shape[1], X.shape[1])
        self.W = np.dot(np.linalg.inv(np.dot(X.T, X)+reg_matrix),np.dot(X.T, Y))

    def predict(self, X):
        return np.dot(X, self.W.T)
