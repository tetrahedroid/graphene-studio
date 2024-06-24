import numpy as np
from dataclasses import dataclass
from typing import Tuple


class SurfaceFunc:
    def __init__(self):
        pass

    def surface(r: np.ndarray) -> float:
        pass

    def gradient(r: np.ndarray) -> np.ndarray:
        pass

    def exfield_potential(self, r):
        Ug = self.surface(r)
        return np.sum(Ug**2)

    def exfield_gradient(self, r):
        Ug = self.surface(r)
        return 2 * self.gradient(r) * Ug[:, np.newaxis]


@dataclass
class Ticks:
    min: float
    binw: float


@dataclass
class Grid:
    values: np.ndarray
    xticks: Ticks
    yticks: Ticks
    zticks: Ticks


def bin(x: np.ndarray, ticks: Ticks) -> Tuple[np.ndarray, np.ndarray]:
    scaled = (x - ticks.min) / ticks.binw
    b = np.floor(scaled)
    return b.astype(int), scaled - b


class GridSurfaceFunc(SurfaceFunc):
    def __init__(self, grid: Grid):
        self.grid = grid

    def surface(self, fracs: np.ndarray) -> np.ndarray:
        pp = fracs - np.floor(fracs + 0.5)
        x = pp[:, 0]
        y = pp[:, 1]
        z = pp[:, 2]

        xbin, xfrac = bin(x, self.grid.xticks)
        ybin, yfrac = bin(y, self.grid.yticks)
        zbin, zfrac = bin(z, self.grid.zticks)

        G = self.grid.values

        return (
            G[xbin, ybin, zbin] * (1 - xfrac) * (1 - yfrac) * (1 - zfrac)
            + G[xbin + 1, ybin, zbin] * xfrac * (1 - yfrac) * (1 - zfrac)
            + G[xbin, ybin + 1, zbin] * (1 - xfrac) * yfrac * (1 - zfrac)
            + G[xbin + 1, ybin + 1, zbin] * xfrac * yfrac * (1 - zfrac)
            + G[xbin, ybin, zbin + 1] * (1 - xfrac) * (1 - yfrac) * zfrac
            + G[xbin + 1, ybin, zbin + 1] * xfrac * (1 - yfrac) * zfrac
            + G[xbin, ybin + 1, zbin + 1] * (1 - xfrac) * yfrac * zfrac
            + G[xbin + 1, ybin + 1, zbin + 1] * xfrac * yfrac * zfrac
        )

    def gradient(self, fracs: np.ndarray) -> np.ndarray:
        pp = fracs - np.floor(fracs + 0.5)
        x = pp[:, 0]
        y = pp[:, 1]
        z = pp[:, 2]

        xbin, xfrac = bin(x, self.grid.xticks)
        ybin, yfrac = bin(y, self.grid.yticks)
        zbin, zfrac = bin(z, self.grid.zticks)

        G = self.grid.values

        gradx = (
            (
                G[xbin + 1, ybin, zbin] * (1 - yfrac) * (1 - zfrac)
                + G[xbin + 1, ybin + 1, zbin] * yfrac * (1 - zfrac)
                + G[xbin + 1, ybin, zbin + 1] * (1 - yfrac) * zfrac
                + G[xbin + 1, ybin + 1, zbin + 1] * yfrac * zfrac
            )
            - (
                G[xbin, ybin, zbin] * (1 - yfrac) * (1 - zfrac)
                + G[xbin, ybin + 1, zbin] * yfrac * (1 - zfrac)
                + G[xbin, ybin, zbin + 1] * (1 - yfrac) * zfrac
                + G[xbin, ybin + 1, zbin + 1] * yfrac * zfrac
            )
        ) / self.grid.xticks.binw
        grady = (
            (
                G[xbin + 1, ybin + 1, zbin] * xfrac * (1 - zfrac)
                + G[xbin + 1, ybin + 1, zbin + 1] * xfrac * zfrac
                + G[xbin, ybin + 1, zbin] * (1 - xfrac) * (1 - zfrac)
                + G[xbin, ybin + 1, zbin + 1] * (1 - xfrac) * zfrac
            )
            - (
                G[xbin + 1, ybin, zbin] * xfrac * (1 - zfrac)
                + G[xbin + 1, ybin, zbin + 1] * xfrac * zfrac
                + G[xbin, ybin, zbin] * (1 - xfrac) * (1 - zfrac)
                + G[xbin, ybin, zbin + 1] * (1 - xfrac) * zfrac
            )
        ) / self.grid.yticks.binw
        gradz = (
            (
                G[xbin + 1, ybin, zbin + 1] * xfrac * (1 - yfrac)
                + G[xbin + 1, ybin + 1, zbin + 1] * xfrac * yfrac
                + G[xbin, ybin, zbin + 1] * (1 - xfrac) * (1 - yfrac)
                + G[xbin, ybin + 1, zbin + 1] * (1 - xfrac) * yfrac
            )
            - (
                G[xbin, ybin, zbin] * (1 - xfrac) * (1 - yfrac)
                + G[xbin, ybin + 1, zbin] * (1 - xfrac) * yfrac
                + G[xbin + 1, ybin, zbin] * xfrac * (1 - yfrac)
                + G[xbin + 1, ybin + 1, zbin] * xfrac * yfrac
            )
        ) / self.grid.zticks.binw

        return np.array([gradx, grady, gradz]).T
