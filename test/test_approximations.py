import numpy as np
from src.approximations import RandomFourierFeatures, NystromApproximation


def test_random_fourier_features():
    X = np.random.rand(10, 5)  # Original data
    rff = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf', random_state=42)
    rff.fit(X)
    X_transformed = rff.transform(X)  # Transformed data

    # Check that the shape is as expected
    assert X_transformed.shape == (10, 10), f"Expected transformed shape (10, 10), got {X_transformed.shape}"

    # Ensure the transformed features are not all zeros
    assert not np.all(X_transformed == 0), "Transformed features should not be all zeros"

    # Check that the transformation changes the features meaningfully
    assert np.std(X_transformed) > 0, "Transformed features should have variability"


def test_nystrom_approximation():
    X = np.random.rand(20, 5)
    nystrom = NystromApproximation(n_components=5, kernel='rbf', gamma=0.5, random_state=42)
    X_transformed = nystrom.fit_transform(X)

    assert X_transformed.shape == (20, 5)
    assert not np.allclose(X, X_transformed), "Transformed features should differ from input"


def test_random_fourier_features_repeatability():
    X = np.random.rand(10, 5)
    rff1 = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf', random_state=42)
    rff2 = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf', random_state=42)

    X_transformed1 = rff1.fit_transform(X)
    X_transformed2 = rff2.fit_transform(X)

    assert np.allclose(X_transformed1, X_transformed2), "Transformations with the same seed should be identical"


def test_nystrom_approximation_repeatability():
    X = np.random.rand(20, 5)
    nystrom1 = NystromApproximation(n_components=5, kernel='rbf', gamma=0.5, random_state=42)
    nystrom2 = NystromApproximation(n_components=5, kernel='rbf', gamma=0.5, random_state=42)

    X_transformed1 = nystrom1.fit_transform(X)
    X_transformed2 = nystrom2.fit_transform(X)

    assert np.allclose(X_transformed1, X_transformed2), "Transformations with the same seed should be identical"


def test_nystrom_approximation_increasing_components():
    X = np.random.rand(50, 10)
    nystrom1 = NystromApproximation(n_components=5, kernel='rbf', gamma=0.5, random_state=42)
    nystrom2 = NystromApproximation(n_components=10, kernel='rbf', gamma=0.5, random_state=42)

    X_transformed1 = nystrom1.fit_transform(X)
    X_transformed2 = nystrom2.fit_transform(X)

    assert X_transformed2.shape[1] > X_transformed1.shape[1], "Increasing components should increase transformed dimensionality"


def test_random_fourier_features_different_kernels():
    X = np.random.rand(10, 5)
    rff_rbf = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='rbf', random_state=42)
    rff_laplace = RandomFourierFeatures(n_components=10, gamma=1.0, kernel='laplace', random_state=42)

    X_transformed_rbf = rff_rbf.fit_transform(X)
    X_transformed_laplace = rff_laplace.fit_transform(X)

    assert not np.allclose(X_transformed_rbf, X_transformed_laplace), "Different kernels should produce different transformations"
