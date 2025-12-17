#Data pipeline
import numpy as np
from tensorflow import keras

def load_mnist(normalize=True, add_channel_dim=True):
    """
    Returns:
        x_train, x_test as float32 arrays.
        Shapes: (N, 28, 28, 1) if add_channel_dim=True, else (N, 28, 28)
    """
    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    if normalize:
        x_train /= 255.0
        x_test  /= 255.0

    if add_channel_dim:
        x_train = x_train[..., None]  # (N, 28, 28, 1)
        x_test  = x_test[..., None]

    return x_train, x_test


def add_gaussian_noise(x, sigma, clip=True, seed=None):
    """
    Adds Gaussian noise N(0, sigma^2) to x.
    Assumes x is in [0,1] if clip=True.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=x.shape).astype(np.float32)
    x_noisy = x + noise

    if clip:
        x_noisy = np.clip(x_noisy, 0.0, 1.0)

    return x_noisy


def make_noisy_pairs(x, sigma, seed=None):
    """
    Creates (x_noisy, x_clean) pairs for denoising.
    """
    x_noisy = add_gaussian_noise(x, sigma=sigma, seed=seed)
    x_clean = x
    return x_noisy, x_clean
