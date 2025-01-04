import numpy as np
from src.utils import GaussianKernel, LaplacianKernel, center_train_gram_matrix, center_test_gram_matrix, LinearKernel, PolynomialKernel, CauchyLorentzKernel


def test_gaussian_kernel():
    kernel = GaussianKernel(sigma=1.0)
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.allclose(Kxx, Kxx.T)


def test_laplacian_kernel():
    kernel = LaplacianKernel(sigma=1.0)
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.allclose(Kxx, Kxx.T)


def test_center_train_gram_matrix():
    Kxx = np.array([[1, 2], [3, 4]])
    Kxx_centered = center_train_gram_matrix(Kxx)
    assert Kxx_centered.shape == Kxx.shape
    assert np.allclose(np.sum(Kxx_centered, axis=0), 0)


def test_center_test_gram_matrix():
    Kxx = np.array([[1, 2], [3, 4]])
    Kxz = np.array([[1, 2], [3, 4]])
    Kxz_centered = center_test_gram_matrix(Kxx, Kxz)
    assert Kxz_centered.shape == Kxz.shape


def test_linear_kernel():
    kernel = LinearKernel()
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.allclose(Kxx, X @ X.T)


def test_polynomial_kernel():
    kernel = PolynomialKernel(degree=2, coef0=1)
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.allclose(Kxx, (X @ X.T + 1) ** 2)


def test_cauchy_lorentz_kernel():
    kernel = CauchyLorentzKernel(gamma=1.0)
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.all(Kxx <= 1)  # Values of Cauchy kernel are between 0 and 1


def test_gaussian_kernel_test_matrix():
    kernel = GaussianKernel(sigma=1.0)
    X = np.array([[1, 2], [3, 4]])
    Z = np.array([[5, 6], [7, 8]])
    Kxz = kernel.get_Kxz(X, Z)
    assert Kxz.shape == (2, 2)
    assert np.all(Kxz >= 0)
