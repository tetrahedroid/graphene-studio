from logging import INFO, basicConfig, getLogger, DEBUG

import numpy as np

from graphenator.graphenate import graphenate
from graphenator import snapshot


def surface(x, cell, R):
    return np.sum(x**2, axis=1) - R**2


def gradient(x, cell):
    return 2 * x


logger = getLogger()
basicConfig(level=INFO)

Npoly = 32
L = 6
cell = np.diag([L, L, L])
R = 2

with open(f"sphere32.yap", "w") as file:
    for x, cell, g in graphenate(
        Npoly,
        lambda x, cell: surface(x, cell, R),
        gradient,
        cell,
        iter=1000,
        cost=1250,  # 250,
        dt=0.05,  # 0.005
        T=0.1,
        repul=4,
    ):
        file.write(snapshot(x, cell, g))
