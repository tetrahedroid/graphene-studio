from typing import Callable
from logging import getLogger
import numpy as np
from scipy.optimize import fmin_cg


def quench_particles(
    r: np.ndarray,
    cell: np.ndarray[3],
    f: Callable[[np.ndarray, np.ndarray], float],
    df: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    """周期境界セル内の粒子のエネルギー最小化。

    Args:
        r (np.ndarray): 粒子のセル相対座標(N x 3)
        cell (np.ndarray[3]): _description_
        f (Callable[[np.ndarray, np.ndarray], float]): ポテンシャルエネルギー関数、引数は粒子のセル相対座標とセル
        df (Callable[[np.ndarray, np.ndarray], np.ndarray]): 勾配関数、引数は粒子のセル相対座標とセル

    Returns:
        _type_: 最小エネルギー構造の粒子のセル相対座標
    """
    logger = getLogger()

    # conjugate gradient minimization
    func = lambda r_linear: f(r_linear.reshape(-1, 3), cell)
    dfunc = lambda r_linear: df(r_linear.reshape(-1, 3), cell).reshape(-1)

    r = fmin_cg(func, r.reshape(-1), fprime=dfunc)
    r = r.reshape(-1, 3)
    # r = fmin_cg(func, r_linear).reshape(-1, 3)

    r -= np.floor(r + 0.5)
    return r
