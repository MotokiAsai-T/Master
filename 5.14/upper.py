from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from model import ModelParams, calc_demand_od, validate_input_params
from utils import calc_MC, recover_m_star

MCCache = Dict[Tuple[int, int, tuple[float, ...]], tuple[float, np.ndarray]]


def _compute_mc_cached(
    v_prices: np.ndarray,
    t: int,
    k: int,
    D_od_n: list[int] | np.ndarray,
    params: ModelParams,
    cache: MCCache,
) -> tuple[float, np.ndarray]:
    v_key = tuple(np.round(v_prices, 8).tolist())
    key = (t, k, v_key)
    if key in cache:
        return cache[key]

    if params.C0 is None or params.C_proc is None or params.S_od is None:
        raise KeyError("C0/C_proc/S_od data not set")

    mc_val, grad = calc_MC(
        v_prices=v_prices,
        D_od_n=D_od_n,
        C0_val=params.C0[t][k],
        C_proc_n=params.C_proc[t][k],
        S_od_n=params.S_od[t][k],
        L=params.Total_vehicles,
        theta_p=params.theta_p,
        theta_c=params.theta_c,
    )
    cache[key] = (mc_val, grad)
    return mc_val, grad


def fista_single_node(
    t: int,
    k: int,
    D_od_n: list[int] | np.ndarray,
    params: ModelParams,
    mc_cache: MCCache,
) -> np.ndarray:
    n_od = params.N_od
    v = np.zeros(n_od, dtype=float)
    v_bar = v.copy()
    t_factor = 1.0
    L_prev = float(params.L0)
    v_hat_prev = v.copy()
    v_hat = v.copy()

    converged = False
    for _iter in range(params.max_iter):
        mc_bar, grad_bar = _compute_mc_cached(v_bar, t, k, D_od_n, params, mc_cache)

        iota = 0
        L_trial = L_prev
        while True:
            L_trial = (params.eta**iota) * L_prev
            v_trial = v_bar + grad_bar / L_trial
            mc_trial, _ = _compute_mc_cached(v_trial, t, k, D_od_n, params, mc_cache)
            q_val = mc_bar + float(np.dot(grad_bar, grad_bar)) / (2.0 * L_trial)
            if mc_trial >= q_val:
                break
            iota += 1
            if iota >= 50:
                break

        L_k = (params.eta**iota) * L_prev
        v_hat = v_bar + grad_bar / L_k
        _, grad_hat = _compute_mc_cached(v_hat, t, k, D_od_n, params, mc_cache)

        if np.linalg.norm(grad_hat) < params.eps_conv:
            converged = True
            break

        if float(np.dot(grad_hat, v_hat - v_hat_prev)) < 0.0:
            t_factor_new = 1.0
        else:
            t_factor_new = (1.0 + math.sqrt(1.0 + 4.0 * t_factor * t_factor)) / 2.0

        v_bar = v_hat + (t_factor - 1.0) / t_factor_new * (v_hat - v_hat_prev)
        v_hat_prev = v_hat.copy()
        t_factor = t_factor_new
        L_prev = L_k

    if not converged:
        print(f"[WARN] FISTA did not converge at node (t={t}, k={k}), using last iterate")

    return v_hat


def solve_upper(
    params: ModelParams,
) -> tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, np.ndarray]]]:
    validate_input_params(params)
    if params.C0 is None or params.C_proc is None or params.S_od is None:
        raise KeyError("C0/C_proc/S_od data not set")

    mc_cache: MCCache = {}
    v_star: Dict[int, Dict[int, np.ndarray]] = {}
    m_star: Dict[int, Dict[int, np.ndarray]] = {}

    for t in range(params.Num_time + 1):
        v_star[t] = {}
        m_star[t] = {}
        for k in range(t + 1):
            D_od_n = calc_demand_od(t, k, params.d0, params.alpha, params.rho or [])
            v_star_tk = fista_single_node(t, k, D_od_n, params, mc_cache)
            m_star_tk = recover_m_star(
                v_star_tk,
                D_od_n,
                params.C0[t][k],
                params.C_proc[t][k],
                params.Total_vehicles,
                params.theta_p,
            )
            v_star[t][k] = v_star_tk
            m_star[t][k] = m_star_tk

            if (not np.all(np.isfinite(v_star_tk))) or (not np.all(np.isfinite(m_star_tk))):
                raise ValueError("Non-finite values detected in v_star/m_star")

    return v_star, m_star
