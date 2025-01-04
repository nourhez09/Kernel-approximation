import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from utils import center_train_gram_matrix, center_test_gram_matrix, GaussianKernel
from approximations import NystromApproximation, RandomFourierFeatures


class KernelRidgeRegression(BaseEstimator, RegressorMixin):
    """
    Kernel Ridge Regression using a kernel provided as a function or class.
    """

    def __init__(self, p=1, lbda=0.1, kernel=None):
        """
        :param p: Number of random Fourier features for the kernel approximation (if needed).
        :param lbda: Regularization parameter (lambda).
        :param kernel: An instance of a kernel class or a kernel function.
        """
        self.p = p
        self.lbda = lbda
        self.kernel = kernel  # The kernel can be a class or a function

    def fit(self, X, y):
        """
        Trains the model by fitting the weights.

        :param X: Feature matrix of shape (n, d).
        :param y: Target vector of shape (n,).
        """
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Compute the kernel matrix (or its approximation)
        if isinstance(self.kernel, RandomFourierFeatures):
            # Use Random Fourier Features for kernel approximation
            Kxx_approx = self.kernel.fit_transform(X)  # Transform data to Fourier feature space
            Kxx = Kxx_approx @ Kxx_approx.T  # Kernel approximation via dot product in feature space
            self.feature_map_ = Kxx_approx
        elif isinstance(self.kernel, NystromApproximation):
            # Use Nyström Approximation for kernel approximation
            Kxx_approx = self.kernel.fit_transform(X)  # Transform data using Nyström
            Kxx = Kxx_approx @ Kxx_approx.T  # Kernel approximation via dot product in feature space
            self.feature_map_ = Kxx_approx
        else:
            # Use exact kernel computation (for example, Gaussian kernel)
            Kxx = self.kernel(X)  

        # Center the kernel matrix using the utility function
        K_centered = center_train_gram_matrix(Kxx)
        # K_centered=Kxx
        # Center the target values
        self.mean_y = np.mean(y)
        y = y - self.mean_y

        # Ridge regularization (solving the normal equation)
        eye = np.eye(K_centered.shape[0])
        self.alpha = np.linalg.solve(K_centered + self.lbda * eye, y)

        # Store training data for prediction
        self.X_train_ = X
        self.kxx_ = Kxx

    def predict(self, X):
        """
        Makes predictions on new data.

        :param X: Feature matrix of shape (n, d).
        :return: Predictions of shape (n,).
        """
        # Convert X to a numpy array
        X = np.array(X)

        # Compute the kernel between new data and training data
        if isinstance(self.kernel, RandomFourierFeatures):
            # Use Random Fourier Features for kernel approximation
            Kxz_approx = self.kernel.transform(X)  # Transform test data to Fourier feature space
            Kxz = self.feature_map_ @ Kxz_approx.T  # Kernel approximation via dot product in feature space
        elif isinstance(self.kernel, NystromApproximation):
            # Use Nyström Approximation for kernel approximation
            Kxz_approx = self.kernel.transform(X)  # Transform test data using Nyström
            Kxz = self.feature_map_ @ Kxz_approx.T  # Kernel approximation via dot product in feature space
        else:
            # Use exact kernel computation 
            Kxz = self.kernel(self.X_train_, X)  

        # Center the new kernel matrix using the same utility function
        K_new_centered = center_test_gram_matrix(self.kxx_, Kxz)

        # Predictions
        y_pred = self.alpha.T@K_new_centered + self.mean_y

        return y_pred


if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Create an RBF kernel with a specific gamma parameter

    # Example dataset
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use Gaussian Kernel
    gaussian_kernel = GaussianKernel(sigma=1.0)
    model = KernelRidgeRegression(kernel=gaussian_kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Use Random Fourier Features with RBF kernel approximation
    rff = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf')
    model_rff = KernelRidgeRegression(kernel=rff)
    model_rff.fit(X_train, y_train)
    y_pred_rff = model_rff.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_rff)
    print(f'Mean Squared Error: {mse}')

    # Use Nyström Approximation
    nystrom = NystromApproximation(n_components=50, gamma=1.0, kernel='rbf')
    model_nystrom = KernelRidgeRegression(kernel=nystrom)
    model_nystrom.fit(X_train, y_train)
    y_pred_nystrom = model_nystrom.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_nystrom)
    print(f'Mean Squared Error: {mse}')
