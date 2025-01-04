import numpy as np
from unittest.mock import patch
from src.visualization import plot_kpca, plot_variance_ratio, plot_silhouette_vs_gamma
from src.utils import explained_variance_ratio, GaussianKernel
from src.tasks import KernelPCA


def test_plot_kpca():
    X = np.random.rand(100, 2)
    X_kpca = np.random.rand(100, 2)
    y = np.random.randint(0, 3, 100)

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_kpca(X, X_kpca, y)
        mock_show.assert_called_once()


def test_plot_variance_ratio():
    eigvals = np.array([4, 3, 2, 1])

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_variance_ratio(eigvals, limit=4)
        mock_show.assert_called_once()


def test_plot_silhouette_vs_gamma():
    X = np.random.rand(100, 2)
    y = np.random.randint(0, 3, 100)

    with patch("matplotlib.pyplot.show") as mock_show:
        plot_silhouette_vs_gamma(
            X,
            y,
            GaussianKernel,
            gammas=np.logspace(-3, 1, 5),
            n_components=2,
            title="Test Silhouette"
        )
        mock_show.assert_called_once()
