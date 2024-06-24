import itertools as it
from logging import getLogger
from typing import Tuple, Callable

import networkx as nx
import numpy as np
import pairlist as pl
import yaplotlib as yap
from cycless import cycles, simplex

from graphenestudio.quench import quench_particles


def firstshell(x: np.ndarray, cell: np.ndarray, rc=None) -> float:
    """
    この関数は、指定されたカットオフ半径内の特定のシステム内の粒子間の平均距離を計算します。

    Args:
      x (np.ndarray): システム内の粒子の位置を表す座標の配列。
      cell (np.ndarray): `firstshell` 関数の `cell` パラメータは、原子が配置されているシミュレーション セルを表しているようです。これは、シミュレーション
    セルの次元を表す numpy 配列であることが予想されます。
      rc: `firstshell` 関数の `rc` パラメータは、結晶構造内の特定の距離内にある隣接原子を識別するためのカットオフ半径を表します。ユーザーが指定しない場合、関数はシミュレーション
    セルのサイズとシステム内の原子の数に基づいて `rc` を計算します。

    Returns:

    関数`firstshell`は、入力numpy配列`x`（粒子の位置）と`cell`（シミュレーションセルの寸法）に基づいて計算された、システムの最初のシェル内の粒子間の平均距離を返します。パラメータ`rc`（カットオフ半径）が指定されていない場合は、シミュレーションセルのサイズとシステム内の粒子の数に基づいて計算されます。関数
    """
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


def snapshot(
    x: np.ndarray, cell: np.ndarray, bondlen: float = 1.2, verbose: bool = True
) -> str:
    """
    この関数は、粒子のシステムの視覚的なスナップショットを、それらの接続と幾何学的特性とともに生成します。

    Args:
      x (np.ndarray): `x` パラメータは、システム内の粒子の位置を表す numpy 配列である必要があります。
      cell (np.ndarray): `snapshot` 関数の `cell` パラメータは、原子が含まれるシミュレーション セルを表します。これは、シミュレーション セルのベクトルを定義する
    3x3 の numpy 配列です。配列の各行は、セルの格子ベクトルを表します。
      bondlen (float):
    `snapshot`関数の`bondlen`パラメータは、システム内の原子間の接続を決定するために使用される結合長を表します。これは、2つの原子が互いに結合しているとみなされる最大距離を指定する浮動小数点値です。この関数では、この距離内の原子のペアは
      verbose (bool):
    `snapshot`関数の`verbose`パラメータは、関数の実行中に追加情報とログメッセージを表示するかどうかを決定するブールフラグです。`verbose`が`True`に設定されている場合、関数は座標、四面体、および.
    Defaults to True

    Returns:

    関数「snapshot」は、入力データ「x」と「cell」の視覚的表現を含む文字列「frame」を返します。文字列「frame」には、セル構造、原子の位置、原子間の結合、配位数、四面体、システム内のサイクルに関する情報が含まれている可能性があります。この関数は、入力パラメータとデータに基づいて原子構造の視覚的なスナップショットを生成するようです。
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


def onestep(
    x: np.ndarray,
    v: np.ndarray,
    cell: np.ndarray,
    f: Callable,
    df: Callable,
    dt: float,
    T: float = None,
    repul: int = 4,
    cost: float = 0,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    この Python 関数 `onestep` は、力とエネルギーの計算に基づいて原子の位置と速度を更新することにより、分子動力学シミュレーションの次のステップを計算します。

    Args:
      x (np.ndarray): `x` パラメータは、シミュレーション内の原子の位置を表す numpy 配列です。配列の各行は、3D 空間内の原子の位置に対応します。
      v (np.ndarray): 関数 `onestep` のパラメータ `v` は、システム内の原子の速度を表します。これは、システム内の各原子の速度を含む配列です。
      cell (np.ndarray):
    提供された関数「onestep」の「cell」パラメータは、原子が閉じ込められているシミュレーションセルを表しているようです。これは、セルマトリックスを表すnumpy配列であることが期待されます。この関数は、分子動力学シミュレーション中に、このシミュレーションセル内の原子の力を計算し、位置と速度を更新します。
      f (Callable): 関数 `f` はシステムの位置エネルギーを表し、関数 `df` はシステム内の原子に作用する力を表します。関数 `onestep`
    のパラメータは、分子動力学シミュレーションで単一の時間ステップを実行するために使用されます。
      df (Callable):
    関数`onestep`の`df`パラメータはCallable型で、原子位置に関するポテンシャルエネルギーの微分を計算する関数を表します。この関数は2つの引数を取ります。セル行列の逆数（`x @
    celli`）でスケールされた原子位置と
      dt (float): `onestep` 関数の `dt` パラメータは、シミュレーションで使用される時間ステップを表します。シミュレーションの各反復で経過する時間を決定します。
      T (float):
    関数`onestep`のパラメータ`T`は、システムの運動エネルギーを制御するための目標温度として使用されます。システムの運動エネルギー（`ek`）が指定された目標温度`T`よりも大きい場合、原子の速度は次の係数で縮小されます。
      repul (int): `onestep` 関数の `repul` パラメータは、反発係数を指定する入力引数として使用されているようです。関数では、デフォルト値の 4
    に設定されています。このパラメータは、システム内の原子間の力や相互作用の計算に影響を与える可能性があります。. Defaults to 4
      cost (float): `onestep` 関数の `cost` パラメータは、ポテンシャル エネルギー (`ep`)
    の計算でスケーリング係数として使用される定数値のようです。これは、ポテンシャル エネルギーを原子の総数 (`Natom`) で割って正規化するために使用されます。. Defaults to 0

    Returns:
      関数 `onestep` は、次の要素を含むタプルを返します:
    1. 更新された位置配列 `x`
    2. 更新された速度配列 `v`
    3. 運動エネルギー `ek`
    4. 位置エネルギー `ep`
    """
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


def random_box(Natom: int) -> np.ndarray:
    """
    この Python 関数は、指定された数の原子に対してボックス内のランダムな座標セットを生成します。

    Args:
      Natom (int):
    `random_box`関数は、指定された数の原子（`Natom`）のボックス内にランダムな座標を生成します。この関数は最初に`Natom`の立方根に基づいて近似グリッドを作成し、次にこのグリッド内にランダムな座標を生成します。最後に、最初の`Natom`をシャッフルして選択します。

    Returns:
      `random_box` 関数は、ボックス内にランダムに配置された原子の座標を含む NumPy 配列を返します。
    """
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


def triangulate(
    Natom: int,
    cell: np.ndarray,
    f: Callable,
    df: Callable,
    dt: float = 0.001,
    T: float = 0.5,
) -> np.ndarray:
    """
    「三角分割」関数は、粒子を急冷し、その後分子動力学を使用して焼き戻しプロセスを実行することで、表面上に一連の点を生成します。

    Args:
      Natom (int): `Natom` パラメータはシステム内の原子の数を表します。これはシミュレーション内に存在する原子の総数を指定する整数値です。
      cell (np.ndarray): `triangulate` 関数の `cell` パラメータは、原子が配置されている結晶格子の単位セルを表します。単位セルの格子ベクトルを定義する NumPy
    配列であることが期待されます。
      f (Callable):
    `triangulate`関数の`f`パラメータはCallable型です。つまり、特定の引数で呼び出して値を返す関数です。この文脈では、`f`はシステムの位置エネルギーを表す関数であると考えられます。次の位置を取ります。
      df (Callable):
    `triangulate`関数の`df`パラメータはCallable型であり、つまり呼び出し可能な関数です。この文脈では、`df`は与えられた関数`f`の勾配を計算する関数であると考えられます。勾配は、
      dt (float):
    `triangulate`関数の`dt`パラメータは、シミュレーションで使用される時間ステップを表します。これは、シミュレーションの各反復中にシステム内の粒子の位置と速度を更新するために使用する時間間隔の小ささを決定します。`dt`値が小さいと、
      T (float): `triangulate` 関数のパラメータ `T` はシステムの温度を表します。これは、三角測量プロセス中に温度を制御するためにシミュレーションで使用されます。
    """
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
