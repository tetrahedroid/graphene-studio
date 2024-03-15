import numpy as np
import pairlist as pl


def firstshell(x, cell, rc=None):
    if rc is None:
        rc = cell[0, 0] / len(x) ** (1 / 3) * 3

    ds = []
    for _, _, d in pl.pairs_iter(x, rc, cell, fractional=False):
        ds.append(d)
    H = np.histogram(ds, bins=30)
    for i in range(len(H[0]) - 1):
        if H[0][i] > H[0][i + 1]:
            break
    return (H[1][i] + H[1][i + 1]) / 2
