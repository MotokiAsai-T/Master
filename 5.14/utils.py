from __future__ import annotations

import math

import numpy as np

try:
    from scipy.special import comb as sp_comb
except Exception:  # pragma: no cover
    sp_comb = None


def calc_pi(t: int, k: int, p: float) -> float:
    if k < 0 or k > t:
        return 0.0
    coeff = float(sp_comb(t, k)) if sp_comb is not None else float(math.comb(t, k))
    return coeff * (p**k) * ((1.0 - p) ** (t - k))


def calc_D(t: int, k: int, d0: int, alpha: int) -> int:
    return int(d0 + alpha * k)


def calc_demand_od(t: int, k: int, d0: int, alpha: int, rho: list[float]) -> list[int]:
    d_total = calc_D(t, k, d0, alpha)
    n_od = len(rho)

    raw = [rho[n] * d_total for n in range(n_od)]
    base = [math.floor(x) for x in raw]
    remainder = d_total - sum(base)
    frac = [raw[n] - base[n] for n in range(n_od)]

    order = sorted(range(n_od), key=lambda n: frac[n], reverse=True)
    for n in order[:remainder]:
        base[n] += 1

    return [int(x) for x in base]


def logit_vehicle(
    v_prices: np.ndarray,
    C0_val: float,
    C_proc_n: np.ndarray,
    theta_p: float,
) -> tuple[float, np.ndarray]:
    utility_wait = -float(C0_val) / theta_p
    utility_proc = -(C_proc_n - v_prices) / theta_p

    max_u = float(max(utility_wait, float(np.max(utility_proc))))
    exp_wait = math.exp(utility_wait - max_u)
    exp_proc = np.exp(utility_proc - max_u)

    z_v = exp_wait + float(np.sum(exp_proc))
    mu_v = -theta_p * (math.log(z_v) + max_u)
    p_v = exp_proc / z_v
    return float(mu_v), p_v.astype(float)


def logit_passenger(v_n: float, S_n: float, theta_c: float) -> tuple[float, float]:
    u_price = -float(v_n) / theta_c
    u_walk = -float(S_n) / theta_c
    max_u = max(u_price, u_walk)

    exp_price = math.exp(u_price - max_u)
    exp_walk = math.exp(u_walk - max_u)
    z_c = exp_price + exp_walk

    v_nc = -theta_c * (math.log(z_c) + max_u)
    p_nc = exp_price / z_c
    return float(v_nc), float(p_nc)


def calc_MC(
    v_prices: np.ndarray,
    D_od_n: list[int] | np.ndarray,
    C0_val: float,
    C_proc_n: np.ndarray,
    S_od_n: np.ndarray,
    L: int,
    theta_p: float,
    theta_c: float,
) -> tuple[float, np.ndarray]:
    demand = np.asarray(D_od_n, dtype=float)
    mu_v, p_v = logit_vehicle(v_prices, C0_val, C_proc_n, theta_p)

    n_od = len(v_prices)
    v_nc = np.zeros(n_od, dtype=float)
    p_nc = np.zeros(n_od, dtype=float)

    for n in range(n_od):
        v_nc[n], p_nc[n] = logit_passenger(float(v_prices[n]), float(S_od_n[n]), theta_c)

    mc = float(L) * mu_v + float(np.dot(demand, v_nc))
    grad = demand * p_nc - float(L) * p_v
    return float(mc), grad.astype(float)


def recover_m_star(
    v_prices: np.ndarray,
    D_od_n: list[int] | np.ndarray,
    C0_val: float,
    C_proc_n: np.ndarray,
    L: int,
    theta_p: float,
) -> np.ndarray:
    demand = np.asarray(D_od_n, dtype=int)
    _, p_v = logit_vehicle(v_prices, C0_val, C_proc_n, theta_p)

    m_star = np.rint(float(L) * p_v).astype(int)
    m_star = np.maximum(0, np.minimum(m_star, demand))

    total = int(np.sum(m_star))
    if total > L:
        print("[WARN] recover_m_star total exceeded L, applying correction")
        overflow = total - L
        order = np.argsort(-m_star)
        for idx in order:
            if overflow <= 0:
                break
            dec = min(int(m_star[idx]), overflow)
            m_star[idx] -= dec
            overflow -= dec

    return m_star.astype(int)
