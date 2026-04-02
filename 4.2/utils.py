from __future__ import annotations

import math
from typing import Dict, List, Tuple

try:
    from scipy.special import comb as sp_comb
    from scipy.special import expit as sp_expit
except Exception:  # pragma: no cover
    sp_comb = None
    sp_expit = None

State = tuple[int, int, int, int, int]


def get_l(s: State, L: int) -> int:
    """状態タプル s=(r1,...,r5) から利用可能車両数 l を返す。"""
    return L - sum(s)


def transition(s: State, u: int, L: int) -> State:
    """行動 u 後の次期状態 s'=(r2,r3,r4,r5,u) を返す。"""
    r1, r2, r3, r4, r5 = s
    _ = r1
    next_s = (r2, r3, r4, r5, int(u))

    assert all(x >= 0 for x in next_s), "next state must be non-negative"
    assert get_l(next_s, L) >= 0, "available vehicles must be non-negative"
    assert sum(next_s) + get_l(next_s, L) == L, "vehicle conservation violated"
    return next_s


def calc_pi(t: int, k: int, p: float) -> float:
    """二項分布ベースの遷移確率 π(t,k) を返す。"""
    if k < 0 or k > t:
        return 0.0

    if sp_comb is not None:
        c = float(sp_comb(t, k))
    else:
        c = float(math.comb(t, k))
    return c * (p**k) * ((1.0 - p) ** (t - k))


def calc_D(t: int, k: int, d0: int, alpha: int) -> int:
    """需要量 D[t][k] を返す。"""
    return int(d0 + alpha * k)


def _sigmoid(x: float) -> float:
    if sp_expit is not None:
        return float(sp_expit(x))
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def z_star(lambda_: float, C_i: List[float], B_i: List[float], theta1: float, theta2: float) -> List[float]:
    """KKT 条件に基づき各リクエストの最適処理確率 z* を返す。"""
    if len(C_i) != len(B_i):
        raise ValueError("C_i and B_i must have same length")

    theta = float(max((theta1 + theta2) / 2.0, 1e-12))
    out: List[float] = []
    for c, b in zip(C_i, B_i):
        z = _sigmoid((b - c + lambda_) / theta)
        out.append(float(min(1.0 - 1e-10, max(1e-10, z))))
    return out


def bisection(
    m: float,
    C_i: List[float],
    B_i: List[float],
    theta1: float,
    theta2: float,
    eps_bisec: float,
) -> float:
    """sum z*(lambda)=m を満たす lambda を二分法で求める。"""
    if not C_i:
        return 0.0

    scale = 100.0 * max(theta1, theta2, 1e-12)
    lambda_min = min(B_i) - min(C_i) - scale
    lambda_max = max(B_i) - max(C_i) + scale

    def f(lmb: float) -> float:
        return sum(z_star(lmb, C_i, B_i, theta1, theta2)) - m

    f_min = f(lambda_min)
    f_max = f(lambda_max)

    if f_min > 0.0 and f_max > 0.0:
        return lambda_min
    if f_min < 0.0 and f_max < 0.0:
        return lambda_max

    mid = (lambda_min + lambda_max) / 2.0
    for _ in range(1000):
        mid = (lambda_min + lambda_max) / 2.0
        f_mid = f(mid)

        if abs(lambda_max - lambda_min) < eps_bisec:
            return mid

        if f_mid == 0.0:
            return mid
        if f_mid * f_min < 0.0:
            lambda_max = mid
        else:
            lambda_min = mid
            f_min = f_mid

    print("[WARN] bisection reached max iterations; returning current midpoint")
    return mid


def _entropy(xs: List[float], theta: float) -> float:
    """Shannon エントロピー項 H(x, theta) = -theta * sum(x log x)。"""
    total = 0.0
    for x in xs:
        xc = min(1.0 - 1e-10, max(1e-10, x))
        total += xc * math.log(xc)
    return -theta * total


def calc_G_bar(
    m: float,
    t: int,
    k: int,
    Cost: Dict[int, Dict[int, List[float]]],
    Penalty: Dict[int, Dict[int, List[float]]],
    d0: int,
    alpha: int,
    p: float,
    theta1: float,
    theta2: float,
    eps_bisec: float,
) -> Tuple[float, float]:
    """期待最小費用 Ḡ_{t,k}(m) と lambda* を返す。"""
    req_n = max(0, calc_D(t, k, d0, alpha))
    c_vec = list(Cost[t][k])[:req_n]
    b_vec = list(Penalty[t][k])[:req_n]

    if req_n == 0 or not c_vec:
        return 0.0, 0.0

    m_clipped = min(max(float(m), 0.0), float(req_n))
    lambda_star = bisection(m_clipped, c_vec, b_vec, theta1, theta2, eps_bisec)
    z = z_star(lambda_star, c_vec, b_vec, theta1, theta2)

    h1 = _entropy(z, theta1)
    one_minus_z = [1.0 - zi for zi in z]
    h2 = _entropy(one_minus_z, theta2)

    g_val = 0.0
    for ci, bi, zi in zip(c_vec, b_vec, z):
        g_val += ci * zi + bi * (1.0 - zi)
    g_val -= h1 + h2

    return calc_pi(t, k, p) * g_val, lambda_star


def finite_diff_dV(
    V_dict: Dict[State, float],
    s: State,
    key: str,
    L: int,
    delta: int = 1,
) -> float:
    """価値関数 V の有限差分近似偏微分を返す。"""
    r1, r2, r3, r4, r5 = s

    def is_valid(state: State) -> bool:
        return all(x >= 0 for x in state) and get_l(state, L) >= 0

    if key == "l":
        s_plus = (r1, r2, r3, r4, r5 - delta)
        s_minus = (r1, r2, r3, r4, r5 + delta)
    elif key == "r5":
        s_plus = (r1, r2, r3, r4, r5 + delta)
        s_minus = (r1, r2, r3, r4, r5 - delta)
    else:
        raise ValueError("key must be 'l' or 'r5'")

    v0 = V_dict.get(s)
    if v0 is None:
        return 0.0

    has_plus = is_valid(s_plus) and (s_plus in V_dict)
    has_minus = is_valid(s_minus) and (s_minus in V_dict)

    if has_plus and has_minus:
        return (V_dict[s_plus] - V_dict[s_minus]) / float(2 * delta)
    if has_plus:
        return (V_dict[s_plus] - v0) / float(delta)
    if has_minus:
        return (v0 - V_dict[s_minus]) / float(delta)
    return 0.0
