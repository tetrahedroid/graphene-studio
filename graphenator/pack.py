import itertools as it
from logging import getLogger

import networkx as nx
import numpy as np
import pairlist as pl
import yaplotlib as yap
from cycless import cycles, simplex

from graphenator.quench import quench_particles


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


def snapshot(x, cell, bondlen=1.2, verbose=True):
    """yaplot化。ついでに配位数を数える。

    Args:
        x (_type_): _description_
        bondlen (_type_, optional): _description_. Defaults to l0*1.2.

    Returns:
        _type_: _description_
    """

    logger = getLogger()

    celli = np.linalg.inv(cell)

    frame = ""

    frame += yap.Layer(1)
    frame += yap.Color(0)
    c = (cell[0] + cell[1] + cell[2]) / 2
    frame += yap.Line(cell[0, :] - c, -c)
    frame += yap.Line(cell[1, :] - c, -c)
    frame += yap.Line(cell[2, :] - c, -c)

    frame += yap.Size(0.05)
    for pos in x:
        frame += yap.Circle(pos)
    nnei = [0] * len(x)

    g = nx.Graph()
    edges = {}
    for i, j, d in pl.pairs_iter(x, bondlen, cell, fractional=False):
        edges[i, j] = 0
        edges[j, i] = 0
        nnei[i] += 1
        nnei[j] += 1

    hist = [0] * 10
    for i in nnei:
        if i > 9:
            i = 0
        hist[i] += 1
    if verbose:
        logger.info(f"{hist} Coords")

    g = nx.Graph(
        [(i, j) for i, j, d in pl.pairs_iter(x, bondlen, cell, fractional=False)]
    )

    frame += yap.Layer(3)
    tetras = [tetra for tetra in simplex.tetrahedra_iter(g)]
    for tetra in tetras:
        for i, j in it.combinations(tetra, 2):
            edges[i, j] = 3
            edges[j, i] = 3
    if verbose:
        logger.info(f"{len(tetras)} Tetras")

    if len(tetras) == 0:
        frame += yap.Layer(4)
        frame += yap.Color(4)
        hist = [0] * 6
        for cycle in cycles.cycles_iter(g, maxsize=5):
            cycle = list(cycle)
            hist[len(cycle)] += 1
            if len(cycle) > 3:
                d = x[cycle] - x[cycle[0]]
                d -= np.floor(d @ celli + 0.5) @ cell
                d += x[cycle[0]]
                frame += yap.Polygon(d)

        if verbose:
            logger.info(f"{hist} Cycles")

    frame += yap.Layer(2)
    for (i, j), palette in edges.items():
        frame += yap.Color(palette)
        d = x[j] - x[i]
        d -= np.floor(d @ celli + 0.5) @ cell
        frame += yap.Line(x[i], x[i] + d)

    frame += yap.NewPage()

    return frame


def onestep(x, v, cell, f, df, dt, T=None, repul=4, cost=0):
    celli = np.linalg.inv(cell)
    Natom = x.shape[0]

    # 座標を半分だけ進める
    x += v * dt / 2
    x -= np.floor(x @ celli + 0.5) @ cell

    # 力を計算する
    F = -df(x @ celli, cell)

    # 速度を進める
    v += F * dt

    # energy monitor
    ek = np.sum(v**2) / 2 / Natom

    # T controller
    if T is not None:
        if ek > T:
            v *= 0.95
        else:
            v *= 1.05

    # 座標を半分だけ進める
    x += v * dt / 2

    ep = f(x @ celli, cell) / Natom
    return x, v, ek, ep


def random_box(
    Natom: int,
) -> np.ndarray:
    logger = getLogger()
    # approx grid
    N3 = int(Natom ** (1 / 3)) + 1

    # Nよりも大きい立方格子
    X, Y, Z = np.mgrid[0:N3, 0:N3, 0:N3] / N3 - 0.5

    # 格子点の座標のリスト
    x = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T

    # 乱数発生
    r = np.random.random(x.shape[0])

    # シャッフルし、最初のNatomを抽出し、絶対座標に変換する
    x = x[np.argsort(r)][:Natom]
    x += np.random.random(x.shape) * 0.01

    return x


# 最適なものを1つだけ返そうと思うから苦労する。
# 次々にyieldして取捨選択はユーザーにまかせよう。


def triangulate(
    Natom: int,
    cell,
    f,
    df,
    dt=0.001,
    T=0.5,
) -> np.ndarray:
    logger = getLogger()

    r = random_box(Natom)

    # まずquenchし、曲面上に点を載せる
    r_quenched = quench_particles(r, cell, f, df)
    x = r_quenched @ cell

    yield x

    v = np.zeros([Natom, 3])

    logger.info("Tempering")
    while True:
        for _ in range(10):
            x, v, _, _ = onestep(x, v, cell, f, df, dt, T=T)

        yield x
