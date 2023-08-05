import numpy as np
from sklearn.datasets import make_classification, make_circles, make_moons

def create_circle_dataset(n_samples=100):
    X, y = make_circles(n_samples=n_samples, noise=0.1, random_state=42)
    return X, y

def create_exclusive_or_dataset(n_samples=100):
    X = np.random.rand(n_samples, 2)
    y = np.logical_xor(X[:, 0] > 0.5, X[:, 1] > 0.5).astype(int)
    return X, y

def create_gaussian_dataset(n_samples=100):
    X, y = make_classification(n_samples=n_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)
    return X, y

def create_spiral_dataset(n_samples=100):
    n = n_samples
    n_class = 2
    n_per_class = n // n_class
    np.random.seed(0)
    X = np.zeros((n_class * n_per_class, 2))
    y = np.zeros(n_class * n_per_class, dtype='uint8')
    for j in range(n_class):
        ix = range(n_per_class * j, n_per_class * (j + 1))
        r = np.linspace(0.0, 1, n_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, n_per_class) + np.random.randn(n_per_class) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y

def create_moon_dataset(n_samples=100):
    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
    return X, y