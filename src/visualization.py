import numpy as np
import matplotlib.pyplot as plt
from utils import explained_variance_ratio, silhouette_score
from tasks import KernelPCA


def plot_kpca(X, X_kpca, y):

    # Visualisation des résultats
    plt.figure(figsize=(12, 6))

    # Données originales
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50)
    plt.title("Données originales")
    plt.xlabel("x1")
    plt.ylabel("x2")

    # Résultat Kernel PCA
    plt.subplot(1, 2, 2)
    plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y, cmap='viridis', s=50)
    plt.title("Projection Kernel PCA")
    plt.xlabel("Composante 1")
    plt.ylabel("Composante 2")

    plt.tight_layout()
    plt.show()


def plot_variance_ratio(eigvals, limit=50):
    # Calcul de la variance expliquée
    explained_variance = explained_variance_ratio(eigvals[:limit])

    # Affichage des résultats
    plt.plot(range(1, len(eigvals[:limit]) + 1), explained_variance, marker='o')
    plt.title("Ratio de variance expliquée cumulée")
    plt.xlabel("Nombre de composantes principales")
    plt.ylabel("Variance expliquée cumulée")
    plt.grid(True)
    plt.show()


def plot_silhouette_vs_gamma(X, y, kernel_class, gammas=None, n_components=2, title="Silhouette Score vs Gamma"):
    """
    Visualizes the silhouette score for different gamma values in Kernel PCA.

    Parameters:
    - X: ndarray of shape (n_samples, n_features), input data.
    - y: ndarray of shape (n_samples,), true labels or cluster assignments.
    - kernel_class: callable, a kernel class like RBFKernel or GaussianKernel.
    - gammas: list or ndarray, range of gamma values to test (default: logspace(-3, 2, 50)).
    - n_components: int, number of components for Kernel PCA (default: 2).
    - title: str, title of the plot (default: "Silhouette Score vs Gamma").

    Returns:
    - None
    """
    if gammas is None:
        gammas = np.logspace(-3, 2, 50)

    silhouette_scores = []

    for g in gammas:
        # Create the kernel instance with the current gamma
        kernel_instance = kernel_class(gamma=g)

        # Initialize KernelPCA with the kernel
        kpca = KernelPCA(n_components=n_components, kernel_func=kernel_instance)

        # Perform Kernel PCA and compute silhouette score
        X_kpca, _, _ = kpca.fit_transform(X)
        score = silhouette_score(X_kpca, y)
        silhouette_scores.append(score)

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(gammas, silhouette_scores, marker='o')
    plt.xscale('log')
    plt.title(title)
    plt.xlabel("Gamma")
    plt.ylabel("Silhouette Score")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    from utils import GaussianKernel
    from sklearn.datasets import make_blobs

    # Generate synthetic data
    X, y = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

    # Apply Kernel PCA and visualize
    gaussian_kernel = GaussianKernel(sigma=1.0)
    kpca = KernelPCA(n_components=2, kernel_func=gaussian_kernel)
    X_kpca, eigvals, _ = kpca.fit_transform(X)

    plot_kpca(X, X_kpca, y)

    # Plot explained variance ratio
    plot_variance_ratio(eigvals, limit=10)

    # Visualize silhouette scores for different gamma values
    plot_silhouette_vs_gamma(X, y, GaussianKernel, gammas=np.logspace(-3, 2, 10), n_components=2)
