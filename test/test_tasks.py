import numpy as np
from src.tasks import KernelRidgeRegression
from src.utils import GaussianKernel, LaplacianKernel
from src.tasks import KernelPCA
from src.approximations import NystromApproximation


def test_kernel_ridge_regression():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = GaussianKernel(gamma=0.5)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (5,), f"Expected predictions of shape (5,), got {predictions.shape}"


def test_kernel_ridge_regression_laplacian():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = LaplacianKernel(gamma=0.5)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (5,), f"Expected predictions of shape (5,), got {predictions.shape}"


def test_kernel_ridge_regression_invalid_kernel():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = lambda X, Y: X @ Y.T  # Invalid kernel function
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)

    try:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        assert predictions.shape == (5,), "Expected predictions to fail with invalid kernel, but succeeded"
    except Exception as e:
        assert "kernel" in str(e), f"Unexpected exception: {e}"


def test_kernel_ridge_regression_centering():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)

    kernel = GaussianKernel(gamma=0.5)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)

    centered_target = y_train - np.mean(y_train)
    assert np.allclose(np.sum(centered_target), 0), "Target values are not properly centered"


def test_kernel_pca_exact():
    def rbf_kernel(X, gamma=0.5):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)
        return K

    X = np.random.rand(10, 5)
    kernel_pca = KernelPCA(n_components=3, kernel_func=rbf_kernel, kernel_params={'gamma': 0.5})
    X_transformed, eigvals, eigvecs = kernel_pca.fit_transform(X)

    assert X_transformed.shape == (10, 3), "Transformed data shape mismatch"
    assert len(eigvals) == 10, "Incorrect number of eigenvalues"
    assert eigvecs.shape == (10, 10), "Eigenvectors shape mismatch"


def test_kernel_pca_nystroem():
    X = np.random.rand(20, 5)
    nystroem_model = NystromApproximation(n_components=10, kernel='rbf', gamma=0.5, random_state=42)
    kernel_pca = KernelPCA(n_components=5)

    X_transformed, eigvals, eigvecs = kernel_pca.fit_transform_approx(X, nystroem_model)

    assert X_transformed.shape == (20, 5), "Transformed data shape mismatch"
    assert len(eigvals) == 20, "Incorrect number of eigenvalues"
    assert eigvecs.shape == (20, 20), "Eigenvectors shape mismatch"


def test_kernel_pca_component_reduction():
    def linear_kernel(X):
        return X @ X.T

    X = np.random.rand(15, 5)
    kernel_pca = KernelPCA(n_components=2, kernel_func=linear_kernel)
    X_transformed, eigvals, eigvecs = kernel_pca.fit_transform(X)

    assert X_transformed.shape == (15, 2), "Component reduction did not work correctly"


def test_kernel_pca_eigenvalue_order():
    def rbf_kernel(X, gamma=1.0):
        n = X.shape[0]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-gamma * np.linalg.norm(X[i] - X[j]) ** 2)
        return K

    X = np.random.rand(12, 3)
    kernel_pca = KernelPCA(n_components=4, kernel_func=rbf_kernel, kernel_params={'gamma': 1.0})
    _, eigvals, _ = kernel_pca.fit_transform(X)

    assert np.all(np.diff(eigvals[::-1]) >= 0), "Eigenvalues are not sorted in descending order"
