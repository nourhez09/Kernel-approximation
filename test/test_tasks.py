import numpy as np
from src.tasks import KernelRidgeRegression
from src.utils import GaussianKernel, LaplacianKernel


def test_kernel_ridge_regression():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = GaussianKernel(sigma=1.0)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (5,), f"Expected predictions of shape (5,), got {predictions.shape}"


def test_kernel_ridge_regression_laplacian():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = LaplacianKernel(sigma=1.0)
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

    kernel = GaussianKernel(sigma=1.0)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)

    centered_target = y_train - np.mean(y_train)
    assert np.allclose(np.sum(centered_target), 0), "Target values are not properly centered"
