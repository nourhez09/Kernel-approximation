import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from utils import center_train_gram_matrix, center_test_gram_matrix, GaussianKernel
from approximations import NystromApproximation, RandomFourierFeatures


class KernelRidgeRegression(BaseEstimator, RegressorMixin):
    """
    Kernel Ridge Regression using a kernel provided as a function or class.
    """

    def __init__(self, p=1, lbda=0.1, kernel=None):
        """
        :param p: Number of random Fourier features for the kernel approximation (if needed).
        :param lbda: Regularization parameter (lambda).
        :param kernel: An instance of a kernel class or a kernel function.
        """
        self.p = p
        self.lbda = lbda
        self.kernel = kernel  # The kernel can be a class or a function

    def fit(self, X, y):
        """
        Trains the model by fitting the weights.

        :param X: Feature matrix of shape (n, d).
        :param y: Target vector of shape (n,).
        """
        # Convert X and y to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Compute the kernel matrix (or its approximation)
        if isinstance(self.kernel, RandomFourierFeatures):
            # Use Random Fourier Features for kernel approximation
            Kxx_approx = self.kernel.fit_transform(X)  # Transform data to Fourier feature space
            Kxx = Kxx_approx @ Kxx_approx.T  # Kernel approximation via dot product in feature space
            self.feature_map_ = Kxx_approx
        elif isinstance(self.kernel, NystromApproximation):
            # Use Nyström Approximation for kernel approximation
            Kxx_approx = self.kernel.fit_transform(X)  # Transform data using Nyström
            Kxx = Kxx_approx @ Kxx_approx.T  # Kernel approximation via dot product in feature space
            self.feature_map_ = Kxx_approx
        elif callable(self.kernel):
            # Use exact kernel computation (for example, Gaussian kernel)
            Kxx = self.kernel(X)
        else:
            Kxx_approx = self.kernel  # Use a precomputed value for Kxx
            Kxx = Kxx_approx @ Kxx_approx.T  # Kernel approximation via dot product in feature space
            self.feature_map_ = Kxx_approx

        # Center the kernel matrix using the utility function
        K_centered = center_train_gram_matrix(Kxx)
        # K_centered=Kxx
        # Center the target values
        self.mean_y = np.mean(y)
        y = y - self.mean_y

        # Ridge regularization (solving the normal equation)
        eye = np.eye(K_centered.shape[0])
        self.alpha = np.linalg.solve(K_centered + self.lbda * eye, y)

        # Store training data for prediction
        self.X_train_ = X
        self.kxx_ = Kxx

    def predict(self, X):
        """
        Makes predictions on new data.

        :param X: Feature matrix of shape (n, d).
        :return: Predictions of shape (n,).
        """
        # Convert X to a numpy array
        X = np.array(X)

        # Compute the kernel between new data and training data
        if isinstance(self.kernel, RandomFourierFeatures):
            # Use Random Fourier Features for kernel approximation
            Kxz_approx = self.kernel.transform(X)  # Transform test data to Fourier feature space
            Kxz = self.feature_map_ @ Kxz_approx.T  # Kernel approximation via dot product in feature space
        elif isinstance(self.kernel, NystromApproximation):
            # Use Nyström Approximation for kernel approximation
            Kxz_approx = self.kernel.transform(X)  # Transform test data using Nyström
            Kxz = self.feature_map_ @ Kxz_approx.T  # Kernel approximation via dot product in feature space
        elif callable(self.kernel):
            # Use exact kernel computation
            Kxz = self.kernel(self.X_train_, X)
        else:
            # Use Nyström Approximation for kernel approximation
            Kxz_approx = X  # pre-transformed test data using Nyström
            Kxz = self.feature_map_ @ Kxz_approx.T  # Kernel approximation via dot product in feature space

        # Center the new kernel matrix using the same utility function
        K_new_centered = center_test_gram_matrix(self.kxx_, Kxz)

        # Predictions
        y_pred = self.alpha.T@K_new_centered + self.mean_y

        return y_pred


class KernelPCA:
    """
    A class for performing Kernel PCA, including an approximate version using Nyström method.
    """

    def __init__(self, n_components, kernel_func=None, kernel_params=None):
        """
        Initialize the KernelPCA object.

        Parameters:
        - n_components: int, number of principal components.
        - kernel_func: callable, kernel function (e.g., RBF kernel).
        - kernel_params: dict, additional parameters for the kernel function.
        """
        self.n_components = n_components
        self.kernel_func = kernel_func
        self.kernel_params = kernel_params if kernel_params else {}

    def fit_transform(self, X):
        """
        Perform Kernel PCA on the input data.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data.

        Returns:
        - X_transformed: ndarray, projected data in the kernel PCA space.
        - eigvals: ndarray, eigenvalues of the centered kernel matrix.
        - eigvecs: ndarray, eigenvectors of the centered kernel matrix.
        """
        # Compute the kernel matrix
        K = self.kernel_func(X, **self.kernel_params)
        # Center the kernel matrix
        K_centered = center_train_gram_matrix(K)
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(K_centered)
        # Sort in descending order
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
        # Select top components
        alphas = eigvecs[:, :self.n_components]
        lambdas = eigvals[:self.n_components]
        # Transform data
        X_transformed = alphas * np.sqrt(lambdas)
        return X_transformed, eigvals, eigvecs

    def fit_transform_approx(self, X, nystroem_model):
        """
        Perform approximate Kernel PCA using the Nyström method.

        Parameters:
        - X: ndarray of shape (n_samples, n_features), input data.
        - nystroem_model: object, a fitted Nyström kernel approximation model.

        Returns:
        - X_transformed: ndarray, projected data in the approximate kernel PCA space.
        - eigvals: ndarray, eigenvalues of the centered kernel matrix.
        - eigvecs: ndarray, eigenvectors of the centered kernel matrix.
        """
        # Fit the Nyström model
        nystroem_model.fit(X)
        # Compute the approximate kernel matrix
        Z = nystroem_model.transform(X)
        K_approx = Z @ Z.T
        # Center the kernel matrix
        K_centered = center_train_gram_matrix(K_approx)
        # Eigen-decomposition
        eigvals, eigvecs = np.linalg.eigh(K_centered)
        # Sort in descending order
        eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]
        # Select top components
        alphas = eigvecs[:, :self.n_components]
        lambdas = eigvals[:self.n_components]
        # Transform data
        X_transformed = alphas * np.sqrt(lambdas)
        return X_transformed, eigvals, eigvecs

class SVC:
    def __init__(self, C=1.0, gamma=0.1):
        self.C = C  # Paramètre de régularisation
        self.gamma = gamma  # Paramètre du noyau RBF
    
    # Fonction noyau RBF
    def rbf_kernel(self, X, Z):
        # Calcul de la matrice de noyau RBF entre X et Z
        pairwise_sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Z**2, axis=1) - 2 * np.dot(X, Z.T)
        return np.exp(-self.gamma * pairwise_sq_dists)
    
    def qp_solver_own(P, q, G, h, A, b, max_iter=1000, tol=1e-6, alpha=1e-2):
        """
        Solves the Quadratic Programming problem using a gradient descent method.
        """
        # Initialize x randomly or at zero
        x = np.zeros(P.shape[1])

        # Gradient descent loop
        for _ in range(max_iter):
            # Calculate the gradient of the objective function
            grad = np.dot(P, x) + q 

            # Gradient descent step
            x = x - alpha * grad

            for i in range(len(G)):
                if np.dot(G[i], x) > h[i]:
                    x = x - alpha * (np.dot(G[i], x) - h[i]) * G[i]

            # Project onto the equality constraints: Ax = b
            x = x + np.linalg.lstsq(A, b - np.dot(A, x), rcond=None)[0]

            # Check for convergence (if the gradient is close to zero)
            if np.linalg.norm(grad) < tol:
                break
        return x
    
    # Entraînement du SVM
    def fit(self, X, y):
        m, n = X.shape
        # Calcul de la matrice de noyau K
        K = self.rbf_kernel(X, X)
        
        # Matrice P, q, G, h pour le problème quadratique
        Y = y.reshape(-1, 1)  # Labels sous forme colonne
        P = np.dot(Y, Y.T) * K  # P = Y * K * Y
        q = -np.ones(m)  # q = -1
        
        # Matrices G et h pour les contraintes (0 <= alpha_i <= C)
        G = np.vstack([-np.eye(m), np.eye(m)])  # G = [-I; I]
        h = np.hstack([np.zeros(m), np.ones(m) * self.C])  # h = [0; C]
        
        # Matrice A et b pour la contrainte d'égalité Y^T alpha = 0
        A = Y.T
        b = np.zeros(1)

        # Résolution du problème quadratique avec cvxopt
        # P = matrix(P)
        # q = matrix(q)
        # G = matrix(G)
        # h = matrix(h)
        
        # Résolution du problème QP
        #solution = solvers.qp(P, q, G, h, A, b)
        solution = qp_solver_own(P, q, G, h, A, b, max_iter=1000, tol=1e-6, alpha=1e-2)
        print(solution)
        
        self.alpha = solution
        
        # Calcul des vecteurs supports et du biais b
        self.support_vectors = X[self.alpha > 1e-5]
        self.support_labels = y[self.alpha > 1e-5]
        self.alpha_sv = self.alpha[self.alpha > 1e-5]
        
        # Calcul du biais b
        b_values = []
        for i in range(len(self.support_vectors)):
            b_values.append(self.support_labels[i] - np.sum(self.alpha_sv * self.support_labels * K[self.alpha > 1e-5, i]))
        self.b = np.mean(b_values)

    # Prédiction avec le SVM
    def predict(self, X):
        # Calcul des noyaux entre les points de test et les vecteurs supports
        K_test = self.rbf_kernel(X, self.support_vectors)
        # Calcul des valeurs de décision
        decision_values = np.dot(self.alpha_sv * self.support_labels, K_test.T) + self.b
        return np.sign(decision_values)  # Prédiction : +1 ou -1



if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Create an RBF kernel with a specific gamma parameter

    # Example dataset
    X = np.random.randn(100, 5)
    y = np.random.randn(100)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use Gaussian Kernel
    gaussian_kernel = GaussianKernel(gamma=0.5)
    model = KernelRidgeRegression(kernel=gaussian_kernel)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Use Random Fourier Features with RBF kernel approximation
    rff = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf')
    model_rff = KernelRidgeRegression(kernel=rff)
    model_rff.fit(X_train, y_train)
    y_pred_rff = model_rff.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_rff)
    print(f'Mean Squared Error: {mse}')

    # Use Nyström Approximation
    nystrom = NystromApproximation(n_components=50, gamma=1.0, kernel='rbf')
    model_nystrom = KernelRidgeRegression(kernel=nystrom)
    model_nystrom.fit(X_train, y_train)
    y_pred_nystrom = model_nystrom.predict(X_test)
    mse = mean_squared_error(y_test, y_pred_nystrom)
    print(f'Mean Squared Error: {mse}')
