import itertools as it
from logging import getLogger

import networkx as nx
import numpy as np
import pairlist as pl
import yaplotlib as yap
from cycless import cycles, simplex


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

    frame += yap.Size(0.2)
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


def force(x, cell, surface, gradient, repul=2, a=4, cost=250, rc=5):
    """
    反発のみにして、長さの尺度を消す。
    """
    logger = getLogger()

    celli = np.linalg.inv(cell)
    Natom = x.shape[0]
    neis = [{} for i in range(Natom)]
    interactions = dict()
    for i, j, d in pl.pairs_iter(x, rc, cell, fractional=False):
        neis[i][j] = d
        neis[j][i] = d

    for i in range(Natom):
        argnei = sorted(neis[i], key=lambda j: neis[i][j])
        for j in argnei[:8]:
            interactions[i, j] = "A"
            interactions[j, i] = "A"

    F = np.zeros([Natom, 3])
    for i, j in interactions:
        if i < j:
            d = x[i] - x[j]
            d -= np.floor(d @ celli + 0.5) @ cell
            r = (d @ d) ** 0.5
            assert r != 0, (i, j, x[i], x[j])
            e = d / r

            # repulsion
            f = -repul * e * a / r ** (repul + 1)
            F[i] -= f
            F[j] += f
    # assert False
    # logger.info(f"{E} PE(1)")

    # さらに、Gyroidの外場を加える。
    Ug = surface(x, cell)
    fg = 2 * cost * gradient(x, cell) * Ug[:, np.newaxis]
    # Dg = 250  # force const
    F -= fg
    # logger.info(f"{np.sum(Ug**2) * Dg} PE(2)")
    return F


def potential_energy(
    x, cell, surface, repul=2, a=4, cost=250, rc=5, return_total=False
):
    """
    反発のみにして、長さの尺度を消す。
    """
    logger = getLogger()

    celli = np.linalg.inv(cell)
    Natom = x.shape[0]
    neis = [{} for i in range(Natom)]
    interactions = set()
    for i, j, d in pl.pairs_iter(x, rc, cell, fractional=False):
        neis[i][j] = d
        neis[j][i] = d

    for i in range(Natom):
        argnei = sorted(neis[i], key=lambda j: neis[i][j])
        for j in argnei[:8]:
            interactions.add((i, j))
            interactions.add((j, i))

    E = 0
    for i, j in interactions:
        if i < j:
            d = x[i] - x[j]
            d -= np.floor(d @ celli + 0.5) @ cell
            r = (d @ d) ** 0.5
            assert r != 0, (i, j, x[i], x[j])

            # repulsion
            E += a / r**repul

    Ei = E

    # さらに、Gyroidの外場を加える。
    Ug = surface(x, cell)
    # logger.info(f"{np.sum(Ug**2) * Dg} PE(2)")
    Ex = cost * np.sum(Ug**2)

    if return_total:
        return Ei + Ex
    return Ei, Ex


def onestep(x, v, cell, surface, gradient, dt, T=None, repul=4, cost=0):
    celli = np.linalg.inv(cell)
    Natom = x.shape[0]

    # 座標を半分だけ進める
    x += v * dt / 2
    x -= np.floor(x @ celli + 0.5) @ cell

    # 力を計算する
    F = force(x, cell, surface, gradient, repul=repul, cost=cost)

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

    Ei, Eg = potential_energy(x, cell, surface, repul=repul, cost=cost)
    return x, v, ek, (Ei + Eg) / Natom


from scipy.optimize import fmin_cg


def quench(x, cell, surface, gradient, repul=4, cost=250):
    logger = getLogger()

    # conjugate gradient minimization
    x = fmin_cg(
        lambda x: potential_energy(
            x.reshape(-1, 3), cell, surface, repul=repul, cost=cost, return_total=True
        ),
        x.reshape(-1),
        fprime=lambda x: -force(
            x.reshape(-1, 3), cell, surface, gradient, repul=repul, cost=cost
        ).reshape(-1),
    ).reshape(-1, 3)

    celli = np.linalg.inv(cell)
    x -= np.floor(x @ celli + 0.5) @ cell
    return x


# 最適なものを1つだけ返そうと思うから苦労する。
# 次々にyieldして取捨選択はユーザーにまかせよう。


def triangulate(
    Natom: int,
    surface,
    gradient,
    cell,
    cost=0,
    dt=0.001,
    T=0.5,
    repul=4,
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
    x = x[np.argsort(r)][:Natom] @ cell

    # 揺らぎを加えて対称性を崩す
    x += np.random.random(x.shape) * 0.01

    # まずquenchし、曲面上に点を載せる
    x = quench(x, cell, surface, gradient, repul=repul, cost=cost)

    yield x

    v = np.zeros([Natom, 3])

    # # debug用: エネルギー保存則を確認する。
    # import matplotlib.pyplot as plt

    # Eps = []
    # Eks = []
    # for loop in range(1000):
    #     logger.debug(f"Adiabat {loop}")
    #     x, v, Ek, Ep = onestep(
    #         x, v, cell, surface, gradient, dt, T=None, cost=cost, repul=repul
    #     )
    #     Eps.append(Ep)
    #     Eks.append(Ek)
    #     # print(Ep, Ek, Ep + Ek)
    # Eps = np.array(Eps)
    # Eks = np.array(Eks)
    # plt.plot(Eps)
    # plt.plot(Eks)
    # plt.plot(Eps + Eks)
    # plt.show()
    # assert False

    logger.info("Tempering")
    while True:
        for _ in range(10):
            x, v, _, _ = onestep(
                x, v, cell, surface, gradient, dt, T=T, cost=cost, repul=repul
            )

        yield x
