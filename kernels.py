import numpy as np

class RBFKernel:
    """Radial Basis Function (Gaussian) Kernel."""
    def __init__(self, lengthscale=1.0, variance=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)

        diff = X[:, None, :] - Y[None, :, :]
        sqdist = np.sum(diff**2, axis=-1)

        return self.variance * np.exp(-0.5 * sqdist / (self.lengthscale**2))

class PeriodicKernel:
    """Periodic Kernel."""
    def __init__(self, lengthscale=1.0, variance=1.0, period=1.0):
        self.lengthscale = lengthscale
        self.variance = variance
        self.period = period
    def __call__(self, X, Y=None):
        X = np.atleast_2d(X)
        Y = X if Y is None else np.atleast_2d(Y)

        diff = X[:, None, :] - Y[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)

        sine_arg = np.pi * dist / self.period
        sin_term = np.sin(sine_arg)
        return self.variance * np.exp(-2 * (sin_term**2) / (self.lengthscale**2))

class SumKernel:
    """Combined Kernel as sum of multiple kernel components (e.g., RBF + Periodic)."""
    def __init__(self, *kernels):
        self.kernels = kernels
    def __call__(self, X, Y=None):
        # Sum the covariance matrices from each component kernel
        K_total = 0
        for kern in self.kernels:
            K_comp = kern(X, Y)
            K_total = K_comp if isinstance(K_total, int) else K_total + K_comp
        return K_total