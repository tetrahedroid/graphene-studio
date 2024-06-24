import numpy as np

# import pairlist as pl
import yaplotlib as yap
from cycless import cycles
from logging import getLogger
import graphenator.pack as pack
import graphenator.graph as graph


def draw_yaplot(x, cell, g):

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
    logger.info(f"{hist} Cycles in the graph")

    return frame


def graphenate0(
    Natom: int,
    surface,
    gradient,
    cell,
    T=0.5,
    dt=0.005,
    cost=250,
    repul=4,
) -> np.ndarray:

    logger = getLogger()
    # make base trianglated surface
    for x in pack.triangulate0(
        Natom,
        surface,
        gradient,
        cell,
        dt=dt,
        T=T,
        cost=cost,
        repul=repul,
    ):
        # すべて三角格子になるように辺を追加する。
        x = pack.quench(x, cell, surface, gradient, repul=repul, cost=cost)
        g_fix = graph.fix_graph(x, cell)

        if g_fix is not None:
            # analyze the triangular adjacency and make the adjacency graph
            triangle_positions, g_adjacency = graph.dual(x, cell, g_fix)
            triangle_positions = graph.quench(triangle_positions, cell, g_adjacency)
            yield triangle_positions, cell, g_adjacency


def graphenate(
    Natom: int,
    cell,
    f,
    df,
    T=0.5,
    dt=0.005,
) -> np.ndarray:

    logger = getLogger()
    # make base trianglated surface
    for x in pack.triangulate(
        Natom,
        cell,
        f,
        df,
        dt=dt,
        T=T,
    ):
        logger.info("packing candid.")
        # すべて三角格子になるように辺を追加する。
        g_fix = graph.fix_graph(x, cell)

        if g_fix is not None:
            # analyze the triangular adjacency and make the adjacency graph
            triangle_positions, g_adjacency = graph.dual(x, cell, g_fix)
            logger.info("Quenching the graph...")
            triangle_positions = graph.quench(triangle_positions, cell, g_adjacency)
            logger.info("Done.")
            yield triangle_positions, cell, g_adjacency


# def is_cubic_cell(cell):
#     Lx, Ly, Lz = cell[0, 0], cell[1, 1], cell[2, 2]
#     return Lx == Ly == Lz and np.count_nonzero(cell - np.diag(np.diagonal(cell))) == 0
