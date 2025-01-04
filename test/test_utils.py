import numpy as np
from src.utils import GaussianKernel, LaplacianKernel, center_train_gram_matrix, center_test_gram_matrix


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