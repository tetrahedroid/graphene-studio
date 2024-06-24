from logging import getLogger
from typing import Callable

import numpy as np
import networkx as nx
import yaplotlib as yap
from cycless import cycles

import graphenestudio.pack as pack
import graphenestudio.graph as graph
from gromacs import write_gro


def draw_yaplot(x: np.ndarray, cell: np.ndarray, g: nx.Graph) -> str:
    """原子の座標とセルとグラフを読みこんでyaplotに変換する。

    Args:
        x (np.ndarray): atomic positions
        cell (np.ndarray): cell matrix
        g (nx.Graph): connectivity graph

    Returns:
        str: A frame of Yaplot
    """
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

    frame += yap.Size(0.05)
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
    logger.info(f"{hist[3:]} 3-8 cycles in the graph")

    return frame


def graphenate(
    Natom: int,
    cell: np.ndarray,
    f: Callable,
    df: Callable,
    T: float = 0.5,
    dt: float = 0.005,
):
    """力場を読みこみ、曲面グラフェンを生成する。

    Args:
        Natom (int): 原子数(斥力粒子の)
        cell (np.ndarray): セル行列
        f (Callable): 力場のポテンシャル
        df (Callable): 力場のポテンシャルの勾配
        T (float, optional): 温度. Defaults to 0.5.
        dt (float, optional): 時間刻み. Defaults to 0.005.

    Yields:
        原子位置, セル行列, 結合グラフ: 炭素の配置と結合
    """
    # make base trianglated surface
    for x in pack.triangulate(
        Natom,
        cell,
        f,
        df,
        dt=dt,
        T=T,
    ):
        # すべて三角格子になるように辺を追加する。
        g_fix = graph.repair_graph(x, cell)

        if g_fix is not None:
            # analyze the triangular adjacency and make the adjacency graph
            triangle_positions, g_adjacency = graph.dual(x, cell, g_fix)
            triangle_positions = graph.quench(triangle_positions, cell, g_adjacency)
            yield triangle_positions, cell, g_adjacency


def dump_gro(x, cell, g, gro):

    # セルの逆行列
    celli = np.linalg.inv(cell)

    # セル相対座標
    r = x @ celli

    # 0-1範囲にする。
    r -= np.floor(r)

    # 絶対座標もそれにあわせる。
    x = r @ cell

    # 1. 辺の平均長を調べ、それがグラフェンのC-C結合長に一致するように構造をスケールする。
    L = []
    for i, j in g.edges():
        d = r[i] - r[j]
        d -= np.floor(d + 0.5)
        d = d @ cell
        L.append((d @ d) ** 0.5)

    Lavg = np.mean(L)

    CC = 0.143 * 0.98  # nm; すこし短かくしておく。

    x_scaled = x * CC / Lavg
    cell_scaled = cell * CC / Lavg

    # 原子の座標を準備する。

    Natom = x_scaled.shape[0]
    frame = {
        "resi_id": np.array([999 for i in range(Natom)]),
        "residue": np.array(["GRPH" for i in range(Natom)]),
        "atom": np.array(["C" for i in range(Natom)]),
        "atom_id": np.array([999 for i in range(Natom)]),
        "position": x_scaled,
        "cell": cell_scaled,
    }

    # 環を数える

    hist = [0] * 9
    for cycle in cycles.cycles_iter(g, maxsize=8):
        cycle = list(cycle)
        hist[len(cycle)] += 1

    write_gro(frame, gro, f"{hist[3:]} 3-8 cycles in the graph")
