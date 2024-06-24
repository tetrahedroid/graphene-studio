import itertools as it
from logging import getLogger

import networkx as nx
import numpy as np
import pairlist as pl
from cycless import cycles, simplex

from graphenator.pack import firstshell


def to_graph(x: np.ndarray, cell: np.ndarray, bondlen: float = 1.2):
    """
    関数 `to_graph` は、入力配列 `x` と `cell`、および結合長 `bondlen` を受け取り、指定された結合長に基づいて `x` 内のポイント間の接続を表すグラフを構築します。

    Args:
      x (np.ndarray): 空間内の点の座標を含む配列。
      cell (np.ndarray): `to_graph` 関数の `cell`
    パラメータは、グラフ内のポイントが配置されているセルの寸法または境界を表しているようです。グラフ構造の周期性または境界を決定するために使用されると考えられます。
      bondlen (float): `to_graph` 関数の `bondlen` パラメータは、グラフ内でどの原子ペアが接続されているとみなされるかを決定するための結合長のしきい値を表します。2
    つの原子間の距離が `bondlen` 以下の場合、グラフ表現では結合していると見なされます。

    Returns:
      関数 `to_graph` は、入力 numpy 配列 `x`、セル配列 `cell`、結合長 `bondlen` を使用して作成されたグラフ
    オブジェクトを返します。この関数は、最初に入力パラメータに基づいてノードのペアとその距離を生成します。次に、NetworkX
    ライブラリを使用してグラフを作成し、四面体と各四面体の最長の辺を含む特定の基準に基づいて辺を削除します。
    """

    # logger = getLogger()

    pairs = {(i, j): d for i, j, d in pl.pairs_iter(x, bondlen, cell, fractional=False)}

    g = nx.Graph(list(pairs))

    remove = set()
    for tetra in simplex.tetrahedra_iter(g):
        subset = {}
        for i, j in it.combinations(tetra, 2):
            if (i, j) in pairs:
                subset[i, j] = pairs[i, j]
            else:
                subset[i, j] = pairs[j, i]
        keys = list(subset.keys())
        values = list(subset.values())
        longest = np.argmax(values)
        # print(keys)
        # print(values)
        # print(longest, subset[keys[longest]])

        remove.add(keys[longest])

    for edge in remove:
        g.remove_edge(*edge)

    return g


def cycle_histogram(g: np.ndarray) -> list:
    """
    この関数は、指定されたグラフ内のサイクル長のヒストグラムを計算します。

    Args:
      g (np.ndarray): 関数 `cycle_histogram` は、numpy 配列 `g` を入力として受け取り、`g`
    で表されるグラフ内のサイクル長のヒストグラムを計算するように設計されているようです。

    Returns:
      関数 `cycle_histogram` は、入力グラフ `g` のサイクル長のヒストグラムを表すリストを返します。リスト内の各要素は、グラフ内の特定の長さのサイクルの数に対応します。
    """
    hist = [0] * 7
    for cycle in cycles.cycles_iter(g, maxsize=6):
        hist[len(cycle)] += 1
    return hist


def repair_graph(x: np.ndarray, cell: np.ndarray) -> nx.Graph:
    """
    この関数は、グラフ内のサイクルに関連する特定の条件に基づいてエッジを追加することでグラフを修復します。

    Args:
      x (np.ndarray): `x` パラメータは、空間内の点の座標を表す NumPy 配列である必要があります。これは、グラフ構築に関連する計算を実行するために `repair_graph`
    関数で使用されます。
      cell (np.ndarray): `repair_graph` 関数の `cell` パラメータは、結晶構造の単位格子を表しているようです。これは、3
    次元空間における単位格子の寸法と角度を定義する numpy 配列であると考えられます。`cell` パラメータは、関数内のさまざまな計算で使用されます。たとえば、

    Returns:
      関数 `repair_graph` は NetworkX グラフ オブジェクトを返します。
    """
    logger = getLogger()

    rpeak = firstshell(x, cell)
    celli = np.linalg.inv(cell)

    g = to_graph(x, cell, rpeak * 1.33)

    logger.info(f"{cycle_histogram(g)[3:]} 3-6 cycles in the packing")

    newedges = []
    for cycle in cycles.cycles_iter(g, maxsize=6):
        if len(cycle) not in (3, 4):
            # 大穴がある場合はあきらめる
            return None

        if len(cycle) == 4:
            # connect the shorter diagonal
            d1 = x[cycle[2]] - x[cycle[0]]
            d1 -= np.floor(d1 @ celli + 0.5) @ cell
            d2 = x[cycle[3]] - x[cycle[1]]
            d2 -= np.floor(d2 @ celli + 0.5) @ cell

            if d1 @ d1 > d2 @ d2:
                newedges.append((cycle[1], cycle[3]))
            else:
                newedges.append((cycle[0], cycle[2]))

    for edge in newedges:
        g.add_edge(*edge)

    return g


def dual(x: np.ndarray, cell: np.ndarray, g: nx.Graph) -> Tuple[np.ndarray, nx.Graph]:
    """
    この関数は、グラフ内の三角形の位置を計算し、共有エッジに基づいて隣接グラフを作成します。

    Args:
      x (np.ndarray): メッシュ内の頂点の座標を表す点の配列。
      cell (np.ndarray): `dual` 関数の `cell` パラメータは、セル情報を含む numpy 配列を表します。この配列は、関数内のセル マトリックス `celli`
    の逆を計算するために使用されます。セル マトリックスは、グラフ内の三角形の位置に関する計算に使用されます。
      g (nx.Graph): 関数 `dual` のパラメータ `g` は、NetworkX
    ライブラリを使用して表現されたグラフです。これは、グラフ内の三角形を反復処理し、三角形に基づいて特定の操作を実行するために使用されるようです。

    Returns:
      関数 `dual` は、次の 2 つの要素を含むタプルを返します:
    1. 入力グラフ `g` 内の三角形の重心を表す位置の配列 `tripos`。
    2. 共有エッジに基づいて三角形間の隣接関係を表す新しいグラフ `adj`。
    """
    celli = np.linalg.inv(cell)
    tripos = []
    edgeowners = dict()
    adj = nx.Graph()
    for o, tri in enumerate(simplex.triangles_iter(g)):
        d = x[list(tri)] - x[tri[0]]
        d -= np.floor(d @ celli + 0.5) @ cell
        c = np.mean(d, axis=0) + x[tri[0]]
        tripos.append(c)

        i, j, k = tri
        if (i, j) in edgeowners:
            adj.add_edge(o, edgeowners[i, j])
        edgeowners[i, j] = o
        edgeowners[j, i] = o

        if (j, k) in edgeowners:
            adj.add_edge(o, edgeowners[j, k])
        edgeowners[j, k] = o
        edgeowners[k, j] = o

        if (i, k) in edgeowners:
            adj.add_edge(o, edgeowners[i, k])
        edgeowners[i, k] = o
        edgeowners[k, i] = o

    return np.array(tripos), adj


def force(x: np.ndarray, cell: np.ndarray, g: nx.Graph):
    """
    この関数は、指定された位置、セル パラメーター、およびグラフ接続を使用して、周期境界条件における粒子システムの力とエネルギーを計算します。

    Args:
      x (np.ndarray): システム内の各粒子の位置の配列。
      cell (np.ndarray): `cell` パラメータは、3 次元の結晶格子の単位セルを定義する numpy
    配列を表しているようです。これは、格子内の粒子間の力を計算する関数で使用されます。
      g (nx.Graph): 関数 `force` のパラメータ `g` は `nx.Graph` 型であり、これはおそらく Python の NetworkX
    ライブラリのグラフのインスタンスです。このパラメータは、分析対象のシステム内のノード間の接続を定義するグラフ構造を表します。

    Returns:
      関数 `force` は 2 つの値を返します。`F` はシステム内の各原子に作用する力を表す numpy 配列であり、`E` はシステムの総エネルギーを表すスカラー値です。
    """
    celli = np.linalg.inv(cell)
    N = x.shape[0]
    F = np.zeros_like(x)
    E = 0
    for i, j in g.edges():
        d = x[i] - x[j]
        d -= np.floor(d @ celli + 0.5) @ cell
        r = np.linalg.norm(d)
        e = d / r
        F[i] -= e * r
        F[j] += e * r
        E += r**2 / 2
    return F, E


def quench(
    x: np.ndarray, cell: np.ndarray, g: nx.Graph, dt: float = 0.01
) -> np.ndarray:
    """
    関数 `quench` は、関数 `force` を使用して計算された力に基づいて、システム内の粒子の位置を繰り返し更新します。

    Args:
      x (np.ndarray): システム内の粒子の位置を表す np.ndarray。
      cell (np.ndarray): `cell` パラメータに関する情報を提供し忘れたようです。`quench` 関数で `cell`
    パラメータが何を表しているかについて、詳細を提供または明確にしていただけますか?
      g (nx.Graph): nx.Graph は、グラフ データ構造を表す NetworkX ライブラリのクラスです。これは、Python
    でグラフを作成および操作するためによく使用されます。このコンテキストでは、変数 `g` は、何らかの計算や操作のために `quench` 関数で使用されているグラフを表している可能性があります。
      dt (float):
    `quench`関数の`dt`パラメータは、シミュレーションで使用される時間ステップを表します。これは、シミュレーションループの各反復間の経過時間を決定します。この場合、`dt`のデフォルト値は0.01に設定されており、これは、

    Returns:
      関数 `quench` は、`force` 関数を使用して計算された力に基づいて位置が更新されるループを 100 回繰り返した後、更新された位置配列 `x` を返します。
    """
    logger = getLogger()
    for loop in range(100):
        F, E = force(x, cell, g)
        # logger.info(E)
        x += F * dt
    return x
