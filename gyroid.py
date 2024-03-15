from logging import INFO, basicConfig, getLogger, DEBUG

import numpy as np

from graphenator.graphenate import graphenate
from graphenator import snapshot


def surface(x, cell, A):
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    assert Lx == Ly and Lx == Lz, "Should be a cubic cell."
    assert (
        np.count_nonzero(cell - np.diag(np.diagonal(cell))) == 0
    ), "Should be an orthogonal cell."

    rx, ry, rz = (2 * np.pi * x / Lx).T
    return (
        np.sin(rx) * np.cos(ry) + np.sin(ry) * np.cos(rz) + np.sin(rz) * np.cos(rx) + A
    )


def gradient(x, cell):
    Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
    assert Lx == Ly and Lx == Lz, "Should be a cubic cell."
    assert (
        np.count_nonzero(cell - np.diag(np.diagonal(cell))) == 0
    ), "Should be an orthogonal cell."

    rx, ry, rz = (2 * np.pi * x / Lx).T
    res = (
        (-np.sin(rx) * np.sin(rz) + np.cos(rx) * np.cos(ry)) * 2 * np.pi / Lx,
        (-np.sin(rx) * np.sin(ry) + np.cos(ry) * np.cos(rz)) * 2 * np.pi / Ly,
        (-np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz)) * 2 * np.pi / Lz,
    )
    return np.array(res).T


logger = getLogger()
basicConfig(
    level=INFO,
    format="[%(asctime)s] [%(process)d] [%(pathname)s:%(lineno)s] [%(levelname)s] %(message)s",
)
# basicConfig(level=DEBUG)

Npoly = 192
Lmagic = 1.0  # 粒子数とサイズの関係
# この計算は自動化せねば。
L = Npoly**0.5 / Lmagic
cell = np.diag([L, L, L])
eccentricity = 0.5  # eccentricity of the gyroid

with open(f"gyroid.yap", "w") as file:
    for x, cell, g in graphenate(
        Npoly,
        lambda x, cell: surface(x, cell, eccentricity),
        gradient,
        cell,
        iter=1000,
        cost=1250,  # 250,
        dt=0.05,  # 0.005
        T=0.1,
        repul=4,
    ):
        file.write(snapshot(x, cell, g))
