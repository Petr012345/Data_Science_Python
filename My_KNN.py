import numpy as np

# My implementation of k-nearest neighbours algorithm


def norm(X):
    for i in X:
        d = sum(X[i]**2)**0.5
        X[i] = X[i]/d
    return X


class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                dists[i][j] = sum((X[i]-self.X_train[j])**2)**0.5
        return dists

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            dists[i] = X
        return dists

    def compute_cosine_distance(self, X):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        num_pix = X.shape[1]
        X = norm(X)
        self.X_train = np.transpose(self.X_train)
        X = np.transpose(X)
        self.X_train = np.reshape(self.X_train, (num_pix, 1, num_train))
        X = np.reshape(X, (num_pix, num_test, 1))
        dists = self.X_train * X
        dists = sum(dists)
        return dists


    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        num_pix = X.shape[1]
        dists = np.zeros((num_test, num_train))
        dists = np.sqrt(-2 * np.dot(X, self.X_train.T) +
                        np.sum(np.square(self.X_train), axis=1) +
                        np.sum(np.square(X), axis=1)[:, np.newaxis])
        return dists

    def no_loops_for_one_shit(self, X):
        dists = np.sqrt(np.sum(np.square(X - self.X_train), axis=1))
        return dists

    def row_prediction_for1(self, dists, k=1):
        dists = np.argsort(dists)
        closest_y = self.y_train[dists[:k]]
        counter = np.zeros(10)
        for j in closest_y:
            counter[j] += 1
        return counter

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        dists = np.argsort(dists)
        for i in range(num_test):
            closest_y = self.y_train[dists[i][:k]]
            counter = np.zeros(10)
            for j in closest_y:
                counter[j] += 1
            y_pred[i] = np.argsort(counter)[-1]
        return y_pred
