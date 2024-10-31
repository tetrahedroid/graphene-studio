from logging import getLogger
from typing import Callable
import itertools as it

import numpy as np
import networkx as nx
import yaplotlib as yap
from cycless import cycles

import graphenestudio.pack as pack
import graphenestudio.graph as graph
from graphenestudio.gromacs import Frame


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


def dump_gro(x, cell, g, file):

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
    frame = Frame(
        resi_id=np.array([1 for i in range(Natom)]),
        residue=np.array(["GRPH" for i in range(Natom)]),
        atom=np.array(["C" for i in range(Natom)]),
        atom_id=np.array([i + 1 for i in range(Natom)]),
        position=x_scaled,
        cell=cell_scaled,
    )

    # 環を数える

    hist = [0] * 9
    for cycle in cycles.cycles_iter(g, maxsize=8):
        cycle = list(cycle)
        hist[len(cycle)] += 1

    frame.write_gro(file, f"{hist[3:]} 3-8 cycles in the graph")


def moleculetype_section(mol_name):
    s = "[ moleculetype ]\n"
    s += "; Name            nrexcl\n"
    s += f"{mol_name}                 3\n\n"
    return s


def atoms_section(N):
    s = "[ atoms ]\n"
    s += ";   nr       type  resnr residue  atom   cgnr     charge       mass  typeB    chargeB      massB\n"
    for i in range(N):
        s += f"{i+1:4}      CJ      1   GRPH      C      {+1:4}       0     12.011\n"
    return s + "\n"


def bonds_section(g):
    s = "[ bonds ]\n"
    s += ";  ai    aj funct            c0            c1            c2            c3\n"
    for i, j in g.edges():
        s += f"{i+1:5} {j+1:5}    1\n"
    return s + "\n"


def pairs_section(g):
    s = "[ pairs ]\n"
    s += ";  ai    aj funct            c0            c1            c2            c3\n"
    for i in g:
        nei0 = set([i])
        nei1 = set(g.neighbors(i))
        nei2 = set([k for j in nei1 for k in g.neighbors(j)]) - nei1 - nei0
        nei3 = set([k for j in nei2 for k in g.neighbors(j)]) - nei2 - nei1 - nei0
        for j in nei3:
            if i < j:
                s += f"{i+1:5} {j+1:5}    1\n"
    return s + "\n"


def angles_section(g):
    s = "[ angles ]\n"
    s += ";  ai    aj    ak funct            c0            c1            c2            c3\n"
    for i in g:
        for j, k in it.combinations(g.neighbors(i), 2):
            s += f"{j+1:5} {i+1:5} {k+1:5}    1\n"
    return s + "\n"


def dihedrals_section(g):
    s = "[ dihedrals ]\n"
    s += ";  ai    aj    ak funct            c0            c1            c2            c3\n"
    for i, j in g.edges():
        ni = set(g.neighbors(i)) - set([j])
        nj = set(g.neighbors(j)) - set([i])
        if len(ni) > 0 and len(nj) > 0:
            ii = np.random.choice(list(ni), 1)[0]
            jj = np.random.choice(list(nj), 1)[0]
            s += f"{ii+1:5} {i+1:5} {j+1:5} {jj+1:5}    3\n"
    return s + "\n"


def generate_top(
    x: np.ndarray, cell: np.ndarray, g: nx.Graph, generate_dihed_list: bool = False
) -> str:
    logger = getLogger()

    celli = np.linalg.inv(cell)

    s = moleculetype_section("GRPH")
    s += atoms_section(len(x))
    s += bonds_section(g)
    # s += pairs_section(g)
    s += angles_section(g)
    if generate_dihed_list:
        s += dihedrals_section(g)
    return s


def _replicate(x: np.ndarray, cell: np.ndarray, g: nx.Graph, direc: int):
    # セルの逆行列
    celli = np.linalg.inv(cell)

    Natom = x.shape[0]

    # セル相対座標
    r = x @ celli

    # 0-1範囲にする。
    r -= np.floor(r)

    # 結合を倍にする。
    newg = nx.Graph()
    for i, j in g.edges():
        d = r[j] - r[i]
        wrap = np.floor(d + 0.5)
        if wrap[direc] != 0:
            # i-jはcellをまたいでいる。
            newg.add_edge(i, j + Natom)
            newg.add_edge(i + Natom, j)
        else:
            newg.add_edge(i, j)
            newg.add_edge(i + Natom, j + Natom)

    # 座標を倍にする
    r1 = r.copy()
    r1[:, direc] += 1
    newr = np.vstack([r, r1])
    newr[:, direc] /= 2
    newcell = cell.copy()
    newcell[direc, :] *= 2

    # 絶対座標に戻す。
    newx = newr @ newcell
    return newx, newcell, newg


def replicate_x(x: np.ndarray, cell: np.ndarray, g: nx.Graph):
    return _replicate(x, cell, g, direc=0)


def replicate_y(x: np.ndarray, cell: np.ndarray, g: nx.Graph):
    return _replicate(x, cell, g, direc=1)


def replicate_z(x: np.ndarray, cell: np.ndarray, g: nx.Graph):
    return _replicate(x, cell, g, direc=2)


def extend_z(x: np.ndarray, cell: np.ndarray, g: nx.Graph):
    # セルの逆行列
    celli = np.linalg.inv(cell)

    # セル相対座標
    r = x @ celli

    # 0-1範囲にする。
    r -= np.floor(r)

    # z方向にセルをまたぐ結合を削る。
    newg = nx.Graph()
    for i, j in g.edges():
        d = r[j] - r[i]
        wrap = np.floor(d + 0.5)
        if wrap[2] == 0:
            newg.add_edge(i, j)

    newcell = cell.copy()
    newcell[2, :] *= 2

    return r @ cell, newcell, newg
