import numpy as np
from src.utils import GaussianKernel, LaplacianKernel, center_train_gram_matrix, center_test_gram_matrix, LinearKernel, PolynomialKernel, CauchyLorentzKernel
from src.utils import explained_variance_ratio, silhouete_score_diff, entropy_of_variance, linear_classification_score


def test_gaussian_kernel():
    kernel = GaussianKernel(gamma=0.5)
    X = np.array([[1, 2], [3, 4]])
    Kxx = kernel.get_Kxx(X)
    assert Kxx.shape == (2, 2)
    assert np.allclose(Kxx, Kxx.T)


def test_laplacian_kernel():
    kernel = LaplacianKernel(gamma=0.5)
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
    kernel = GaussianKernel(gamma=0.5)
    X = np.array([[1, 2], [3, 4]])
    Z = np.array([[5, 6], [7, 8]])
    Kxz = kernel.get_Kxz(X, Z)
    assert Kxz.shape == (2, 2)
    assert np.all(Kxz >= 0)


def test_explained_variance_ratio():
    eigvals = np.array([4, 3, 2, 1])
    evr = explained_variance_ratio(eigvals)

    assert len(evr) == len(eigvals), "Explained variance ratio length mismatch"
    assert np.isclose(evr[-1], 1.0), "Final explained variance ratio should be 1"


def test_silhouete_score_diff():
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=100, n_features=2, centers=3, random_state=42)
    X_kpca = X + np.random.normal(0, 0.1, X.shape)  # Simulate a Kernel PCA transformation

    sil_original, sil_kpca = silhouete_score_diff(X, X_kpca, y)

    assert sil_original >= 0, "Original silhouette score should be non-negative"
    assert sil_kpca >= 0, "Kernel PCA silhouette score should be non-negative"


def test_entropy_of_variance():
    evr = np.array([0.5, 0.3, 0.2])
    entropy = entropy_of_variance(evr)

    assert entropy > 0, "Entropy should be positive"
    assert np.isclose(entropy, -np.sum(evr * np.log(evr + 1e-10))), "Entropy calculation mismatch"


def test_linear_classification_score():
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    accuracy = linear_classification_score(X, y, test_size=0.3, random_state=42)

    assert 0 <= accuracy <= 1, "Accuracy should be between 0 and 1"
    assert isinstance(accuracy, float), "Accuracy should be a float"
