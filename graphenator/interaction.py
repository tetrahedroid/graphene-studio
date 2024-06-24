from logging import getLogger
import numpy as np
import pairlist as pl


def repulsive_force(
    r: np.ndarray, cell: np.ndarray, repul: int = 2, a: float = 4, rc: float = 5
) -> np.ndarray:
    """
    この関数は、システム内の原子の位置と指定された反発パラメータに基づいて、原子間の反発力を計算します。

    Args:
      r (np.ndarray): パラメータ `r` は、システム内の原子の位置を表す numpy 配列です。配列の各行は、3D 空間内の単一の原子の位置に対応します。
      cell (np.ndarray):
    `repulsive_force`関数の`cell`パラメータは、原子が配置されるシミュレーションセルを表します。これは、シミュレーションセルの寸法と方向を定義するnumpy配列です。セルパラメータは、シミュレーションセル内の原子の位置を計算し、相互作用を計算するために使用されます。
      repul (int): `repulsive_force` 関数の `repul` パラメータは、反発力の計算で距離を何倍にするかを表します。これは、原子間の反発力を計算する式で使用されます。.
    Defaults to 2
      a (float): `repulsive_force` 関数のパラメータ `a` は、原子間の反発力を計算する際に使用される定数のようです。これは、`f = -repul * e * a /
    distance ** (repul + 1)` として力の大きさを計算する際に使用されます。. Defaults to 4
      rc (float): `repulsive_force` 関数の `rc`
    パラメータは、カットオフ半径を表します。これは、原子間の相互作用を考慮するための距離しきい値を定義するために使用されます。このカットオフ半径よりも近い原子のペアは、相互作用の計算で考慮されます。.
    Defaults to 5

    Returns:
      関数 `repulsive_force` は、システム内の各原子に作用する反発力を表す NumPy 配列 `F` を返します。
    """
    celli = np.linalg.inv(cell)
    Natom = r.shape[0]
    neis = [{} for i in range(Natom)]
    interactions = dict()
    for i, j, distance in pl.pairs_iter(r, rc, cell):
        neis[i][j] = distance
        neis[j][i] = distance

    for i in range(Natom):
        argnei = sorted(neis[i], key=lambda j: neis[i][j])
        for j in argnei[:8]:
            interactions[i, j] = "A"
            interactions[j, i] = "A"

    x = r @ cell
    F = np.zeros([Natom, 3])
    for i, j in interactions:
        if i < j:
            D = x[i] - x[j]
            D -= np.floor(D @ celli + 0.5) @ cell
            distance = (D @ D) ** 0.5
            assert distance != 0, (i, j, x[i], x[j])
            e = D / distance

            # repulsion
            f = -repul * e * a / distance ** (repul + 1)
            F[i] -= f
            F[j] += f
    # assert False
    # logger.info(f"{E} PE(1)")

    return F


def repulsive_potential(
    r: np.ndarray, cell: np.ndarray, repul: int = 2, a: float = 4, rc: float = 5
) -> float:
    """
    この関数は、原子の位置と指定されたカットオフ距離に基づいて、原子間の反発ポテンシャルエネルギーを計算します。

    Args:
      r (np.ndarray): パラメータ `r` は、空間内の原子の位置を表す numpy 配列です。配列の各行は、3D 空間内の単一の原子の位置に対応します。
      cell (np.ndarray): `repulsive_potential` 関数の `cell` パラメータは、原子が配置されているシミュレーション
    セルを表しているようです。これは、シミュレーション ボックスを定義するセル ベクトルを含む numpy 配列です。この関数は、このシミュレーションで特定の距離内にある原子間の反発ポテンシャル
    エネルギーを計算します。
      repul (int): `repulsive_potential` 関数の `repul`
    パラメータは、反発ポテンシャルエネルギーの計算で原子間の距離がどれだけ増加するかを表します。これは、原子間の反発エネルギーを計算する式 `a / distance**repul` で使用されます。`.
    Defaults to 2
      a (float): `repulsive_potential`関数のパラメータ`a`は、原子間の反発ポテンシャルエネルギーの計算に使用される定数を表します。これは、式`E + = a /
    distance**repul`で使用され、原子ペア間の反発エネルギーをそれらの距離に基づいて計算します。. Defaults to 4
      rc (float): `repulsive_potential` 関数の `rc`
    パラメータはカットオフ半径を表します。これは、反発ポテンシャルエネルギーの計算で原子間の相互作用が考慮されない距離しきい値を定義するために使用されます。. Defaults to 5

    Returns:
      関数 `repulsive_potential` は、入力パラメータ `r` (原子の位置)、`cell` (シミュレーション セルの寸法)、`repul` (反発力)、`a`
    (反発強度)、および `rc` (カットオフ半径) に基づいて計算された総反発ポテンシャル エネルギーを返します。
    """
    celli = np.linalg.inv(cell)
    Natom = r.shape[0]
    neis = [{} for i in range(Natom)]
    interactions = set()
    for i, j, distance in pl.pairs_iter(r, rc, cell):
        neis[i][j] = distance
        neis[j][i] = distance

    for i in range(Natom):
        argnei = sorted(neis[i], key=lambda j: neis[i][j])
        for j in argnei[:8]:
            interactions.add((i, j))
            interactions.add((j, i))

    x = r @ cell
    E = 0
    for i, j in interactions:
        if i < j:
            D = x[i] - x[j]
            D -= np.floor(D @ celli + 0.5) @ cell
            distance = (D @ D) ** 0.5
            assert distance != 0, (i, j, x[i], x[j])

            # repulsion
            E += a / distance**repul

    return E
