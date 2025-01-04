import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Kernel:
    def compute_norm_f(self, Kxx, alpha):
        """
        Computes the norm of the function in the feature space.

        :param Kxx: np.ndarray of shape (n, n) - The Gram matrix (kernel matrix) of the training data.
        :param alpha: np.ndarray of shape (n,) - The dual coefficients.
        :return: float - The computed norm.
        """
        norm_f = np.dot(alpha, np.dot(Kxx, alpha))
        return norm_f

    def compute_prediction_train(self, Kxx, alpha):
        """
        Computes the predictions on the training set.

        :param Kxx: np.ndarray of shape (n, n) - The Gram matrix (kernel matrix) of the training data.
        :param alpha: np.ndarray of shape (n,) - The dual coefficients.
        :return: np.ndarray of shape (n,) - Predictions on the training data.
        """
        y = np.dot(Kxx, alpha)
        return y

    def compute_prediction_test(self, Kxz, alpha):
        """
        Computes the predictions on the test set.

        :param Kxz: np.ndarray of shape (n, m) - The kernel matrix between training and test data.
        :param alpha: np.ndarray of shape (n,) - The dual coefficients.
        :return: np.ndarray of shape (m,) - Predictions on the test data.
        """
        y = np.dot(Kxz.T, alpha)
        return y


class GaussianKernel(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, X, Z=None):
        if Z is None:
            return self.get_Kxx(X)
        else:
            return self.get_Kxz(X, Z)

    def get_Kxx(self, X):
        """
        Computes the Gram matrix (kernel matrix) for the training data using the Gaussian (RBF) kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :return: np.ndarray of shape (n, n) - The kernel matrix.
        """
        n = X.shape[0]
        Kxx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Kxx[i, j] = np.exp(-self.gamma * np.linalg.norm(X[i, :] - X[j, :]) ** 2)
        return Kxx

    def get_Kxz(self, X, Z):
        """
        Computes the kernel matrix between the training data X and the test data Z.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :param Z: np.ndarray of shape (m, d) - The feature matrix for the test data.
        :return: np.ndarray of shape (n, m) - The kernel matrix between X and Z.
        """
        n, m = X.shape[0], Z.shape[0]
        Kxz = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                Kxz[i, j] = np.exp(-self.gamma * np.linalg.norm(X[i, :] - Z[j, :]) ** 2)
        return Kxz


class LinearKernel(Kernel):
    def __init__(self):
        pass

    def get_Kxx(self, X):
        """
        Computes the Gram matrix (kernel matrix) for the training data using the linear kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :return: np.ndarray of shape (n, n) - The kernel matrix.
        """
        Kxx = np.dot(X, X.T)
        return Kxx

    def get_Kxz(self, X, Z):
        """
        Computes the kernel matrix between the training data X and the test data Z using the linear kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :param Z: np.ndarray of shape (m, d) - The feature matrix for the test data.
        :return: np.ndarray of shape (n, m) - The kernel matrix between X and Z.
        """
        Kxz = np.dot(X, Z.T)
        return Kxz


class LaplacianKernel(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def __call__(self, X, Z=None):
        if Z is None:
            return self.get_Kxx(X)  # Compute the Gram matrix for training data
        else:
            return self.get_Kxz(X, Z)  # Compute the kernel matrix between training and test data

    def get_Kxx(self, X):
        """
        Computes the Gram matrix (kernel matrix) for the training data using the Laplacian kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :return: np.ndarray of shape (n, n) - The kernel matrix.
        """
        n = X.shape[0]
        Kxx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Kxx[i, j] = np.exp(-self.gamma * np.linalg.norm(X[i, :] - X[j, :], ord=1))
        return Kxx

    def get_Kxz(self, X, Z):
        """
        Computes the kernel matrix between the training data X and the test data Z using the Laplacian kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :param Z: np.ndarray of shape (m, d) - The feature matrix for the test data.
        :return: np.ndarray of shape (n, m) - The kernel matrix between X and Z.
        """
        n, m = X.shape[0], Z.shape[0]
        Kxz = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                Kxz[i, j] = np.exp(-self.gamma * np.linalg.norm(X[i, :] - Z[j, :], ord=1))
        return Kxz


class CauchyLorentzKernel(Kernel):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def get_Kxx(self, X):
        """
        Computes the Gram matrix (kernel matrix) for the training data using the Cauchy Lorentz kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :return: np.ndarray of shape (n, n) - The kernel matrix.
        """
        n = X.shape[0]
        Kxx = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                Kxx[i, j] = 1 / (1 + np.linalg.norm(X[i, :] - X[j, :]) ** 2 / self.gamma ** 2)
        return Kxx

    def get_Kxz(self, X, Z):
        """
        Computes the kernel matrix between the training data X and the test data Z using the Cauchy Lorentz kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :param Z: np.ndarray of shape (m, d) - The feature matrix for the test data.
        :return: np.ndarray of shape (n, m) - The kernel matrix between X and Z.
        """
        n, m = X.shape[0], Z.shape[0]
        Kxz = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                Kxz[i, j] = 1 / (1 + np.linalg.norm(X[i, :] - Z[j, :]) ** 2 / self.gamma ** 2)
        return Kxz


class PolynomialKernel(Kernel):
    def __init__(self, degree=2, coef0=1):
        self.degree = degree
        self.coef0 = coef0

    def get_Kxx(self, X):
        """
        Computes the Gram matrix (kernel matrix) for the training data using the Polynomial kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :return: np.ndarray of shape (n, n) - The kernel matrix.
        """
        Kxx = (np.dot(X, X.T) + self.coef0) ** self.degree
        return Kxx

    def get_Kxz(self, X, Z):
        """
        Computes the kernel matrix between the training data X and the test data Z using the Polynomial kernel.

        :param X: np.ndarray of shape (n, d) - The feature matrix for the training data.
        :param Z: np.ndarray of shape (m, d) - The feature matrix for the test data.
        :return: np.ndarray of shape (n, m) - The kernel matrix between X and Z.
        """
        Kxz = (np.dot(X, Z.T) + self.coef0) ** self.degree
        return Kxz


def center_train_gram_matrix(Kxx):
    """
    Centers the kernel matrix for the training set.

    :param Kxx: np.ndarray of shape (n, n) - The kernel matrix of the training data.
    :return: np.ndarray of shape (n, n) - The centered kernel matrix.
    """
    n = Kxx.shape[0]
    unit_matrix = np.ones((n, n)) / n
    Kxx_c = Kxx - np.dot(unit_matrix, Kxx) - np.dot(Kxx, unit_matrix) + np.dot(np.dot(unit_matrix, Kxx), unit_matrix)
    return Kxx_c


def center_test_gram_matrix(Kxx, Kxz):
    """
    Centers the kernel matrix for the test set.

    :param Kxx: np.ndarray of shape (n, n) - The kernel matrix of the training data.
    :param Kxz: np.ndarray of shape (n, m) - The kernel matrix between the training and test data.
    :return: np.ndarray of shape (n, m) - The centered kernel matrix between training and test data.
    """
    n, m = Kxz.shape
    unit_matrix_n = np.ones((n, n)) / n
    unit_matrix_m = np.ones((n, m)) / n
    Kxz_c = Kxz - np.dot(unit_matrix_n, Kxz) - np.dot(Kxx, unit_matrix_m) + np.dot(np.dot(unit_matrix_n, Kxx), unit_matrix_m)
    return Kxz_c


def explained_variance_ratio(eigvals):
    total_variance = np.sum(eigvals)
    explained_variance = np.cumsum(eigvals) / total_variance
    return explained_variance


def silhouete_score_diff(X, X_kpca, y):
    # Calcul du score silhouette
    sil_score = silhouette_score(X, y)
    print(f"Score silhouette original: {sil_score:.2f}")
    # Calcul du score silhouette
    sil_score_kpca = silhouette_score(X_kpca, y)
    print(f"Score silhouette après Kernel PCA: {sil_score_kpca:.2f}")
    return sil_score, sil_score_kpca


def entropy_of_variance(explained_variance_ratio):
    normalized_ratios = explained_variance_ratio / np.sum(explained_variance_ratio)
    entropy = -np.sum(normalized_ratios * np.log(normalized_ratios + 1e-10))  # Avoid log(0)
    print(f"Entropy of Variance Ratios: {entropy:.2f}")
    return entropy


def linear_classification_score(X, y, test_size=0.3, random_state=40):
    # Diviser les données projetées
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Appliquer un classifieur sur les données projetées
    clf = SVC(kernel='linear', random_state=random_state)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Calculer la précision
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision linear classification : {accuracy:.2f}")
    return accuracy


if __name__ == '__main__':
    # Example usage of the kernels
    X_train = np.random.rand(5, 3)  # 5 samples, 3 features
    X_test = np.random.rand(3, 3)   # 3 test samples, 3 features

    gaussian_kernel = GaussianKernel(gamma=0.5)
    Kxx_gaussian = gaussian_kernel.get_Kxx(X_train)
    Kxz_gaussian = gaussian_kernel.get_Kxz(X_train, X_test)

    laplacian_kernel = LaplacianKernel(gamma=0.5)
    Kxx_laplacian = laplacian_kernel.get_Kxx(X_train)
    Kxz_laplacian = laplacian_kernel.get_Kxz(X_train, X_test)

    cauchy_kernel = CauchyLorentzKernel(gamma=1.0)
    Kxx_cauchy = cauchy_kernel.get_Kxx(X_train)
    Kxz_cauchy = cauchy_kernel.get_Kxz(X_train, X_test)

    polynomial_kernel = PolynomialKernel(degree=2, coef0=1)
    Kxx_poly = polynomial_kernel.get_Kxx(X_train)
    Kxz_poly = polynomial_kernel.get_Kxz(X_train, X_test)
