import numpy as np
from src.tasks import KernelRidgeRegression
from src.utils import GaussianKernel


def test_kernel_ridge_regression():
    X_train = np.random.rand(10, 2)
    y_train = np.random.rand(10)
    X_test = np.random.rand(5, 2)

    kernel = GaussianKernel(sigma=1.0)
    model = KernelRidgeRegression(lbda=0.1, kernel=kernel)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    assert predictions.shape == (5,), f"Expected predictions of shape (5,), got {predictions.shape}"
