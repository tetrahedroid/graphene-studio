import numpy as np

from graphenator.surface import SurfaceFunc


class Sphere(SurfaceFunc):
    def __init__(self, radius=0.0):
        self.R = radius
        assert 0 < radius < 0.5

    def surface(self, x):
        return np.sum(x**2, axis=1) - self.R**2

    def gradient(self, x):
        return 2 * x
