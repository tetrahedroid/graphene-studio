from logging import getLogger
import numpy as np
import pairlist as pl


def repulsive_force(r, cell, repul=2, a=4, rc=5):
    """
    反発のみにして、長さの尺度を消す。
    """
    logger = getLogger()

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


def repulsive_potential(r, cell, repul=2, a=4, rc=5):
    """
    反発のみにして、長さの尺度を消す。
    """
    logger = getLogger()

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
