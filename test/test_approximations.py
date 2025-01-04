import numpy as np
from src.approximations import RandomFourierFeatures, NystromApproximation


def test_random_fourier_features():
    X = np.random.rand(10, 5)
    rff = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf', random_state=42)
    rff.fit(X)
    X_transformed = rff.transform(X)

    assert X_transformed.shape == (10, 10)


def test_nystrom_approximation():
    X = np.random.rand(20, 5)
    nystrom = NystromApproximation(n_components=5, kernel='rbf', gamma=0.5, random_state=42)
    X_transformed = nystrom.fit_transform(X)

    assert X_transformed.shape == (20, 5)