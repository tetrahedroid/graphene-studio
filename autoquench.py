import sys

from cycless import cycles, simplex
import networkx as nx

import matplotlib.pyplot as plt

import pairlist as pl
import numpy as np
import yaplotlib as yap
import itertools as it
from logging import getLogger, basicConfig, INFO, StreamHandler

logger = getLogger()
basicConfig(level=INFO)
# logger.addHandler(StreamHandler(stream=sys.stdout))
# N = 288  # 一番小さいサイズでもこんなになる。これで自発的に結晶になる気がしない。
# T = 1.0  # temperature

N = int(sys.argv[1])

basename = None
if len(sys.argv) > 2:
    basename = sys.argv[2]

Lmagic = 1.0  # 粒子数とサイズの関係

L = N**0.5 / Lmagic

x = np.random.random([N, 3]) * L
dt = 0.003

# 192粒子でセルサイズ11.3
# セルサイズを2倍にすると面積2倍で粒子数4倍


# 平衡距離
l0 = 1.0  # 短距離
l1 = 3**0.5  # 長距離
# 相互作用のカットオフ距離。全対について計算するわけではないので、この距離は近接対を抽出する目的にのみ使っている。
rc = l1 * 3

# シミュレーションセル(立方体)
cell = np.diag([L, L, L])


def force2(x, n=2, a=4, Dg=250, rc=l1 * 3, plt=None, peak=False):
    """
    反発のみにして、長さの尺度を消す。
    """

    neis = [{} for i in range(N)]
    interactions = dict()
    ds = []
    for i, j, d in pl.pairs_iter(x, rc, cell, fractional=False):
        neis[i][j] = d
        neis[j][i] = d
        ds.append(d)
        # interactions[i, j] = "A"
        # interactions[j, i] = "A"
    H = np.histogram(ds, bins=30)
    for i in range(len(H[0]) - 1):
        if H[0][i] > H[0][i + 1]:
            break
    if peak:
        rpeak = (H[1][i] + H[1][i + 1]) / 2
    if plt is not None:
        H = plt.hist(ds, bins=30, color="gray")
        plt.stem(rpeak, 100)
        plt.show()

    for i in range(N):
        argnei = sorted(neis[i], key=lambda j: neis[i][j])
        for j in argnei[:8]:
            interactions[i, j] = "A"
            interactions[j, i] = "A"

    F = np.zeros([N, 3])
    E = 0
    for i, j in interactions:
        if i < j:
            d = x[i] - x[j]
            d -= np.floor(d / L + 0.5) * L
            r = np.linalg.norm(d)
            e = d / r

            # repulsion
            E += a / r**n
            f = -n * e * a / r ** (n + 1)
            F[i] -= f
            F[j] += f
    # assert False
    # logger.info(f"{E} PE(1)")

    # さらに、Gyroidの外場を加える。
    rx, ry, rz = (2 * np.pi * x / L).T

    Ug = np.sin(rx) * np.cos(ry) + np.sin(ry) * np.cos(rz) + np.sin(rz) * np.cos(rx)
    fx, fy, fz = (
        (-np.sin(rx) * np.sin(rz) + np.cos(rx) * np.cos(ry)) * Ug * 4 * np.pi / L,
        (-np.sin(rx) * np.sin(ry) + np.cos(ry) * np.cos(rz)) * Ug * 4 * np.pi / L,
        (-np.sin(ry) * np.sin(rz) + np.cos(rx) * np.cos(rz)) * Ug * 4 * np.pi / L,
    )
    # Dg = 250  # force const
    F -= np.array([fx, fy, fz]).T * Dg
    # logger.info(f"{np.sum(Ug**2) * Dg} PE(2)")
    E += np.sum(Ug**2) * Dg

    if peak:
        return F, E, rpeak
    return F, E


def snapshot(x, bondlen=l0 * 1.2, verbose=True):
    """yaplot化。ついでに配位数を数える。

    Args:
        x (_type_): _description_
        bondlen (_type_, optional): _description_. Defaults to l0*1.2.

    Returns:
        _type_: _description_
    """

    logger = getLogger()

    frame = ""

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
                d -= np.floor(d / L + 0.5) * L
                d += x[cycle[0]]
                frame += yap.Polygon(d)

        if verbose:
            logger.info(f"{hist} Cycles")

    frame += yap.Layer(2)
    for (i, j), palette in edges.items():
        frame += yap.Color(palette)
        d = x[j] - x[i]
        d -= np.floor(d / L + 0.5) * L
        frame += yap.Line(x[i], x[i] + d)

    frame += yap.NewPage()

    return frame


# Quenching

if basename is not None:
    with open(f"{basename}_{N}.yap", "w") as f:
        f.write("")

for loop in range(1000):
    F, E, rpeak = force2(x, n=4, peak=True, rc=rc)
    rc = rpeak * 2

    # 終了判定
    dx = np.mean(F**2) * 3
    if dx < 0.01:
        break

    if loop % 10 == 0:
        logger.info((loop, E / N, rpeak, dx, dt))
    # print(F)

    # overdamped action
    x += F * dt

    x -= np.floor(x / L + 0.5) * L

    # 長さをどうするかは検討の余地あり。一応、Gyroidの表面積から推定はできるが表面積がわからない。
    if basename is not None:
        with open(f"{basename}_{N}.yap", "a") as f:
            f.write(snapshot(x, rpeak * 1.33, verbose=(loop % 100 == 0)))

print(L, L, L)
print(N)
for pos in x:
    print(*pos)
