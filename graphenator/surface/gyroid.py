import numpy as np

from graphenator.surface import SurfaceFunc


class Gyroid(SurfaceFunc):
    def __init__(self, eccentricity=0.0):
        self.ecc = eccentricity

    def surface(self, r):
        rx, ry, rz = (2 * np.pi * r).T
        return (
            np.sin(rx) * np.cos(ry)
            + np.sin(ry) * np.cos(rz)
            + np.sin(rz) * np.cos(rx)
            + self.ecc
        )

    def gradient(self, r):
        rx, ry, rz = (2 * np.pi * r).T
        res = (
            (-np.sin(rx) * np.sin(rz) + np.cos(rx) * np.cos(ry)) * 2 * np.pi,
            (-np.sin(rx) * np.sin(ry) + np.cos(ry) * np.cos(rz)) * 2 * np.pi,
            (-np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz)) * 2 * np.pi,
        )
        return np.array(res).T
