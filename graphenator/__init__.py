import numpy as np
import pairlist as pl
import yaplotlib as yap
from cycless import cycles
from logging import getLogger


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


def snapshot(x, cell, g):

    logger = getLogger()

    celli = np.linalg.inv(cell)

    frame = ""

    frame += yap.SetPalette(7, 128, 255, 128)
    frame += yap.Layer(1)
    frame += yap.Color(0)
    c = (cell[0] + cell[1] + cell[2]) / 2
    frame += yap.Line(cell[0, :] - c, -c)
    frame += yap.Line(cell[1, :] - c, -c)
    frame += yap.Line(cell[2, :] - c, -c)

    frame += yap.Size(0.2)
    for pos in x:
        frame += yap.Circle(pos)

    hist = [0] * 9
    for cycle in cycles.cycles_iter(g, maxsize=8):
        cycle = list(cycle)
        hist[len(cycle)] += 1
        d = x[cycle] - x[cycle[0]]
        d -= np.floor(d @ celli + 0.5) @ cell
        c = np.mean(d, axis=0) + x[cycle[0]]
        dc = np.floor(c @ celli + 0.5) @ cell
        d = d + x[cycle[0]] - dc
        frame += yap.Layer(len(cycle))
        frame += yap.Color(len(cycle))
        frame += yap.Polygon(d)

    frame += yap.NewPage()
    logger.info(f"{hist} Cycles")

    return frame
