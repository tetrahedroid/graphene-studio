from logging import INFO, basicConfig, getLogger, DEBUG

import numpy as np

from graphenator import draw_yaplot, graphenate, is_cubic_cell


def surface(x, cell, A):
    assert is_cubic_cell(cell)

    L = cell[0, 0]
    rx, ry, rz = (2 * np.pi * x / L).T
    return (
        np.sin(rx) * np.sin(ry) * np.sin(rz)
        + np.cos(rx) * np.cos(ry) * np.sin(rz)
        + np.cos(rx) * np.sin(ry) * np.cos(rz)
        + np.sin(rx) * np.cos(ry) * np.cos(rz)
        + A
    )


def gradient(x, cell):
    assert is_cubic_cell(cell)

    L = cell[0, 0]
    rx, ry, rz = (2 * np.pi * x / L).T

    res = (
        (
            -np.sin(rx) * np.sin(ry) * np.cos(rz)
            - np.sin(rx) * np.cos(ry) * np.sin(rz)
            + np.cos(rx) * np.sin(ry) * np.sin(rz)
            + np.cos(rx) * np.cos(ry) * np.cos(rz)
        )
        * 2
        * np.pi
        / L,
        (
            np.sin(rx) * np.cos(ry) * np.sin(rz)
            - np.cos(rx) * np.sin(ry) * np.sin(rz)
            + np.cos(rx) * np.cos(ry) * np.cos(rz)
            - np.sin(rx) * np.sin(ry) * np.cos(rz)
        )
        * 2
        * np.pi
        / L,
        (
            np.sin(rx) * np.sin(ry) * np.cos(rz)
            + np.cos(rx) * np.cos(ry) * np.cos(rz)
            - np.cos(rx) * np.sin(ry) * np.sin(rz)
            - np.sin(rx) * np.cos(ry) * np.sin(rz)
        )
        * 2
        * np.pi
        / L,
    )
    return np.array(res).T


logger = getLogger()
basicConfig(
    level=INFO,
    format="[%(asctime)s] [%(process)d] [%(pathname)s:%(lineno)s] [%(levelname)s] %(message)s",
)
# basicConfig(level=DEBUG)

Npoly = 300
Lmagic = 1.0  # 粒子数とサイズの関係
# この計算は自動化せねば。
L = Npoly**0.5 / Lmagic
cell = np.diag([L, L, L])
eccentricity = 0.0  # eccentricity of the gyroid

with open(f"diamond.yap", "w") as file:
    count = 100
    for x, cell, g in graphenate(
        Npoly,
        lambda x, cell: surface(x, cell, eccentricity),
        gradient,
        cell,
        cost=1250,  # 250,
        dt=0.02,  # 0.005
        T=0.01,
        repul=4,
    ):
        file.write(draw_yaplot(x, cell, g))
        count -= 1
        if count == 0:
            break
