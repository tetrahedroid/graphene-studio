import numpy as np

from graphenator.surface import SurfaceFunc


class PSurface(SurfaceFunc):
    def __init__(self, eccentricity=0.0):
        self.ecc = eccentricity

    def surface(self, r):
        rx, ry, rz = (2 * np.pi * r).T
        return np.cos(rx) + np.cos(ry) + np.cos(rz) + self.ecc

    def gradient(self, r):
        rx, ry, rz = (2 * np.pi * r).T

        res = (
            -np.sin(rx) * 2 * np.pi,
            -np.sin(ry) * 2 * np.pi,
            -np.sin(rz) * 2 * np.pi,
        )
        return np.array(res).T
