import numpy as np 
import torch 
from sklearn.kernel_approximation import Nystroem
from sklearn.datasets import make_moons
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted, validate_data



class RandomFourierFeatures(BaseEstimator):

    def __init__(self, n_components=100, gamma=1.0, kernel='rbf', random_state=None):
        """
        Initializes the RandomFourierFeatures class.
        
        :param n_components: The number of random Fourier features to generate.
        :param gamma: A kernel parameter (scale of the distance function).
        :param kernel: The kernel type ('rbf', 'laplace', 'cauchy').
        :param random_state: Seed for random number generation.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.random_state = random_state
        self.W = None  # Random weight matrix
        self.b = None  # Random bias vector

    def get_kernel(self):
        """
        Returns the kernel type being used for random Fourier features.
        This determines the distribution from which the random weights are sampled.
        
        :return: A string indicating the kernel type ('rbf', 'laplace', 'cauchy').
        """
        if self.kernel not in ['rbf', 'laplace', 'cauchy']:
            raise ValueError(f"Unsupported kernel: {self.kernel}. Choose from 'rbf', 'laplace', or 'cauchy'.")
        return self.kernel

    def fit(self, X):
        """
        Fits the random Fourier feature model by generating random weights
        and biases based on the chosen kernel type.
        
        :param X: Input data. A 2D array of shape (n_samples, n_features).
        """
        # Initialize the random number generator
        rng = np.random.RandomState(self.random_state)
        X = validate_data(self, X=X, reset=False)
        # Number of samples and features in the input data
        n_samples, n_features = X.shape
        
        # Get the selected kernel type
        kernel_type = self.get_kernel()
        
        # Generate the random weight matrix W based on the kernel type
        if kernel_type == 'rbf':
            # For RBF, W follows a normal distribution
            self.W = rng.normal(0, np.sqrt(2 * self.gamma), (self.n_components, n_features))
        elif kernel_type == 'laplace':
            # For Laplacian kernel, W follows a Laplace distribution
            self.W = rng.laplace(0, 1 / self.gamma, (self.n_components, n_features))
        elif kernel_type == 'cauchy':
            # For Cauchy-Lorentz kernel, W follows a standard Cauchy distribution scaled by gamma
            self.W = rng.standard_cauchy((self.n_components, n_features)) / self.gamma
            # Optionally, clip extreme values of W to prevent numerical instability

        # Generate the random bias vector b (uniformly distributed)
        self.b = rng.uniform(0, 2 * np.pi, self.n_components)

    def transform(self, X):
        """
        Transforms the input data into the random Fourier feature space.
        
        :param X: Input data. A 2D array of shape (n_samples, n_features).
        :return: The transformed data. A 2D array of shape (n_samples, n_components).
        """
        # if self.W is None or self.b is None:
        #     raise RuntimeError("The model has not been fitted yet. Call 'fit' before 'transform'.")
        # check_is_fitted(self)
        # Compute the projections
        X_proj = X @ self.W.T + self.b
        return np.sqrt(2 / self.n_components) * np.cos(X_proj)

    def fit_transform(self, X):
        """
        Fits the model and transforms the input data into the random Fourier feature space in a single step.
        
        param X: Input data. A 2D array of shape (n_samples, n_features).
        return: The transformed data. A 2D array of shape (n_samples, n_components).
        """
        self.fit(X)
        return self.transform(X)


class NystromApproximation(BaseEstimator):
    """
    Nystrom approximation for kernel-based methods.

    This class implements the Nystrom method to approximate a kernel matrix using
    a subset of data points (landmarks) and provides a low-rank approximation
    for large-scale kernel methods.

    Args:
        n_components (int): Number of landmark points to sample.
        kernel (str): Kernel type ('rbf', 'laplace', 'cauchy', 'linear', 'polynomial').
        gamma (float): Kernel parameter controlling the spread of the kernel (used for RBF, Laplace, and Cauchy kernels).
        coef0 (float): Independent term in the polynomial kernel.
        degree (int): Degree of the polynomial kernel.
        random_state (int, optional): Random seed for reproducibility.

    Attributes:
        X_m (numpy.ndarray): Landmark points used for approximation.
        W_inv_sqrt (numpy.ndarray): Inverse square root of the landmark kernel matrix.
    """
    def __init__(self, n_components=100, kernel="rbf", gamma=1.0, coef0=1, degree=3, random_state=None):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.random_state = random_state
        self.X_m = None
        self.W_inv_sqrt = None

    def _get_kernel_function(self):
        """
        Returns the kernel function based on the kernel type.
        """
        if self.kernel == "rbf":
            return lambda X, Y: np.exp(-self.gamma * np.linalg.norm(X[:, None] - Y[None, :], axis=2) ** 2)
        elif self.kernel == "laplace":
            return lambda X, Y: np.exp(-self.gamma * np.linalg.norm(X[:, None] - Y[None, :], axis=2))
        elif self.kernel == "cauchy":
            return lambda X, Y: 1 / (1 + self.gamma * np.linalg.norm(X[:, None] - Y[None, :], axis=2) ** 2)
        elif self.kernel == "linear":
            return lambda X, Y: X @ Y.T
        elif self.kernel == "polynomial":
            return lambda X, Y: (self.gamma * X @ Y.T + self.coef0) ** self.degree
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel}. Choose from 'rbf', 'laplace', 'cauchy', 'linear', or 'polynomial'.")

    def fit(self, X):
        """
        Fits the Nystrom approximation by selecting landmark points and computing
        the inverse square root of the landmark kernel matrix.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            self: Fitted instance of the NystromApproximation class.
        """
        # Initialize random state
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        
        # Randomly sample landmark indices
        indices = rng.choice(n_samples, size=self.n_components, replace=False)
        self.X_m = X[indices]

        # Compute the kernel matrix for the landmark points
        kernel_func = self._get_kernel_function()
        K = kernel_func(self.X_m, self.X_m)  # (n_components, n_components)

        # Compute the inverse square root of the kernel matrix using SVD
        U, S, Vt = np.linalg.svd(K)
        S_inv_sqrt = np.diag(1.0 / np.sqrt(S + 1e-8))  # Add small value for numerical stability
        self.W_inv_sqrt = U @ S_inv_sqrt @ Vt

        return self

    def transform(self, X):
        """
        Transforms the input data into the low-rank approximation space using the
        Nystrom method.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components).
        """
        # if self.X_m is None or self.W_inv_sqrt is None:
        #     raise RuntimeError("The model has not been fitted. Call 'fit' before 'transform'.")
        # check_is_fitted(self)

        # Compute the cross-kernel matrix between X and landmark points
        kernel_func = self._get_kernel_function()
        C = kernel_func(X, self.X_m)  # (n_samples, n_components)

        # Compute the low-rank approximation
        Z = C @ self.W_inv_sqrt  # (n_samples, n_components)

        return Z

    def fit_transform(self, X):
        """
        Fits the model and transforms the input data into the low-rank approximation
        space in a single step.

        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features).

        Returns:
            numpy.ndarray: Transformed data of shape (n_samples, n_components).
        """
        self=self.fit(X)
        return self.transform(X)


if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
    X = StandardScaler().fit_transform(X)  # Standardize features

    # Example with NystromApproximation
    print("=== Nystrom Approximation Example ===")
    nystrom_model = NystromApproximation(n_components=100, kernel="laplace", gamma=0.5, random_state=42)
    X_nystrom = nystrom_model.fit_transform(X)  # Perform Nystrom approximation

    # Fit a logistic regression on the transformed data
    clf_nystrom = LogisticRegression(max_iter=1000, random_state=42)
    clf_nystrom.fit(X_nystrom, y)
    accuracy_nystrom = clf_nystrom.score(X_nystrom, y)
    print(f"Nystrom approximation accuracy: {accuracy_nystrom:.2f}")

        # Example with RandomFourierFeatures
    print("\n=== Random Fourier Features Example ===")
    rff_model = RandomFourierFeatures(n_components=100, kernel="cauchy", gamma=0.5, random_state=42)
    X_rff = rff_model.fit_transform(X)  # Generate random Fourier features

    # Fit a logistic regression on the transformed data
    clf_rff = LogisticRegression(max_iter=1000, random_state=42)
    clf_rff.fit(X_rff, y)
    accuracy_rff = clf_rff.score(X_rff, y)
    print(f"Random Fourier Features accuracy: {accuracy_rff:.2f}")



