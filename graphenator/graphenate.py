import itertools as it
from logging import getLogger
import time

import networkx as nx
import numpy as np
import pairlist as pl
import yaplotlib as yap
from cycless import cycles, simplex

import graphenator.dual as dual
from graphenator import firstshell


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
    fg = cost * gradient(x, cell) * Ug[:, np.newaxis]
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


def quenching(x, cell, surface, gradient, repul=4, cost=250):
    logger = getLogger()

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
    iter=10000,
    cost=0,
    dt=0.001,
    T=0.5,
    file=None,
    repul=4,
) -> np.ndarray:
    logger = getLogger()
    # approx grid
    N3 = int(Natom ** (1 / 3)) + 1
    X, Y, Z = np.mgrid[0:N3, 0:N3, 0:N3] / N3 - 0.5
    x = np.vstack((X.ravel(), Y.ravel(), Z.ravel())).T
    r = np.random.random(x.shape[0])
    x = x[np.argsort(r)][:Natom] @ cell
    x += np.random.random(x.shape) * 0.01

    v = np.zeros([Natom, 3])

    # 初期配置がひどいはずなので、最初だけQuenchする
    # Quenchを適度に行う良い方針はないのか。

    # conjugate gradient minimization
    now = time.time()
    x = quenching(x, cell, surface, gradient, repul=repul, cost=cost)
    logger.info(f"CG quench {time.time()-now} sec")

    if file is not None:
        rpeak = firstshell(x, cell)
        file.write(snapshot(x, cell, rpeak * 1.35))

    # # debug用: エネルギー保存則を確認する。
    # 外場をなしにしても、EkとEp の振幅が倍違うように見える。
    # # import matplotlib.pyplot as plt
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
    for loop in range(iter):
        x, v, Ek, Ep = onestep(
            x, v, cell, surface, gradient, dt, T=T, cost=cost, repul=repul
        )

        rpeak = firstshell(x, cell)

        if loop % 10 == 0:
            logger.info((loop, Ek, Ep))
            if file is not None:
                # xx = x.copy()
                # xx = quenching(xx, cell, surface, gradient, repul=repul, cost=cost)
                file.write(snapshot(x, cell, rpeak * 1.35))

    # 最後にQuench
    logger.info("Quench again")
    now = time.time()
    x = quenching(x, cell, surface, gradient, repul=repul, cost=cost)
    logger.info(f"CG quench {time.time()-now} sec")

    # with open(f"T{T}-2.yap", "a") as f:
    #     f.write(snapshot(x, cell, rpeak * 1.35))

    # with open(f"T{T}-last.x", "w") as f:
    #     print(L, L, L, file=f)
    #     print(N, file=f)
    #     for pos in x:
    #         print(*pos, file=f)
    return x


def graphenate(
    Natom: int,
    surface,
    gradient,
    cell,
    T=0.5,
    dt=0.005,
    iter=10000,
    cost=250,
    progress=None,
    repul=4,
) -> np.ndarray:

    # make base trianglated surface
    x = triangulate(
        Natom,
        surface,
        gradient,
        cell,
        dt=dt,
        T=T,
        iter=iter,
        cost=cost,
        file=progress,
        repul=repul,
    )
    g_fix = dual.fix_graph(x, cell)

    # analyze the triangular adjacency and make the adjacency graph
    triangle_positions, g_adjacency = dual.dualize(x, cell, g_fix)
    triangle_positions = dual.quench(
        triangle_positions, cell, g_adjacency, file=progress
    )


# with open("T0.5-last.x") as f:
#     lines = f.readlines()
#     cell = np.diag([float(x) for x in lines[0].split()])
#     x = []
#     for line in lines[2:]:
#         x.append([float(x) for x in line.split()])
#     x = np.array(x)
