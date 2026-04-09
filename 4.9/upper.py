from __future__ import annotations

import math
from typing import Dict, Set, Tuple

from model import ModelParams, validate_input_params
from utils import calc_D, calc_G_bar, calc_pi, finite_diff_dV, get_l, transition

State = tuple[int, int, int, int, int]
StateSet = Dict[int, Dict[int, Set[State]]]


def _validate_tree_coverage(params: ModelParams) -> None:
    """Cost/Penalty が全ての (t, k) をカバーしていることを確認する。"""
    if params.Cost is None or params.Penalty is None:
        raise ValueError("Cost and Penalty must be provided")

    for t in range(params.Num_time + 1):
        if t not in params.Cost or t not in params.Penalty:
            raise KeyError(f"Missing t={t} in Cost/Penalty")
        for k in range(t + 1):
            if k not in params.Cost[t] or k not in params.Penalty[t]:
                raise KeyError(f"Missing (t={t}, k={k}) in Cost/Penalty")


def forward_pass(Num_time: int, Total_vehicles: int, d0: int, alpha: int) -> StateSet:
    """前向きパスで到達可能状態集合 S[t][k] を構築する。"""
    S: StateSet = {t: {k: set() for k in range(t + 1)} for t in range(Num_time + 1)}
    S[0][0].add((0, 0, 0, 0, 0))

    for t in range(0, Num_time):
        for k in range(0, t + 1):
            demand = max(0, calc_D(t, k, d0, alpha))
            for s in list(S[t][k]):
                l = get_l(s, Total_vehicles)
                u_max = min(l, demand)
                for u in range(0, u_max + 1):
                    s_next = transition(s, u, Total_vehicles)
                    S[t + 1][k].add(s_next)
                    S[t + 1][k + 1].add(s_next)
    return S


def projected_gradient(
    s: State,
    t: int,
    k: int,
    V: Dict[int, Dict[int, Dict[State, float]]],
    dV_dl: Dict[int, Dict[int, Dict[State, float]]],
    dV_dr5: Dict[int, Dict[int, Dict[State, float]]],
    S: StateSet,
    params: ModelParams,
) -> Tuple[int, float, float]:
    """射影勾配降下法で最適行動 u* を計算する。"""
    _ = S
    l = get_l(s, params.Total_vehicles)
    demand = max(0, calc_D(t, k, params.d0, params.alpha))
    u_max = min(l, demand)

    if u_max <= 0:
        g0, lam0 = calc_G_bar(
            0.0,
            t,
            k,
            params.Cost or {},
            params.Penalty or {},
            params.d0,
            params.alpha,
            params.p_inc,
            params.theta1,
            params.theta2,
            params.eps_bisec,
        )
        return 0, g0, lam0

    u = u_max / 2.0
    last_g = 0.0
    last_lambda = 0.0

    converged = False
    for _ in range(1000):
        g_val, lambda_star = calc_G_bar(
            u,
            t,
            k,
            params.Cost or {},
            params.Penalty or {},
            params.d0,
            params.alpha,
            params.p_inc,
            params.theta1,
            params.theta2,
            params.eps_bisec,
        )
        last_g = g_val
        last_lambda = lambda_star

        grad_future = 0.0
        if t < params.Num_time:
            s_int = transition(s, int(round(u)), params.Total_vehicles)

            dl_low = dV_dl.get(t + 1, {}).get(k, {}).get(s_int, 0.0)
            dr5_low = dV_dr5.get(t + 1, {}).get(k, {}).get(s_int, 0.0)
            dl_high = dV_dl.get(t + 1, {}).get(k + 1, {}).get(s_int, 0.0)
            dr5_high = dV_dr5.get(t + 1, {}).get(k + 1, {}).get(s_int, 0.0)

            grad_low = -dl_low + dr5_low
            grad_high = -dl_high + dr5_high

            p_up = params.p_inc
            p_down = 1.0 - p_up
            grad_future = p_down * grad_low + p_up * grad_high

        grad = calc_pi(t, k, params.p_inc) * lambda_star + grad_future

        if abs(grad) < params.eps:
            converged = True
            break

        u = max(0.0, min(float(u_max), u - params.alpha_lr * grad))

    if not converged:
        print("[WARN] projected_gradient reached max iterations; returning current iterate")

    return int(round(u)), last_g, last_lambda


def solve_upper(
    params: ModelParams,
) -> Tuple[
    Dict[int, Dict[int, Dict[State, float]]],
    Dict[int, Dict[int, Dict[State, int]]],
    Dict[int, Dict[int, Dict[State, float]]],
    Dict[int, Dict[int, Dict[State, float]]],
]:
    """上位問題を解き、V・u_opt・有限差分勾配を返す。"""
    validate_input_params(params)
    _validate_tree_coverage(params)

    S = forward_pass(params.Num_time, params.Total_vehicles, params.d0, params.alpha)

    V: Dict[int, Dict[int, Dict[State, float]]] = {}
    u_opt: Dict[int, Dict[int, Dict[State, int]]] = {}
    dV_dl: Dict[int, Dict[int, Dict[State, float]]] = {}
    dV_dr5: Dict[int, Dict[int, Dict[State, float]]] = {}

    for t in range(params.Num_time, -1, -1):
        V[t] = {}
        u_opt[t] = {}
        dV_dl[t] = {}
        dV_dr5[t] = {}

        for k in range(0, t + 1):
            V[t][k] = {}
            u_opt[t][k] = {}
            dV_dl[t][k] = {}
            dV_dr5[t][k] = {}

            for s in S[t][k]:
                u_star, g_val, _ = projected_gradient(s, t, k, V, dV_dl, dV_dr5, S, params)
                stage_value = g_val

                if t < params.Num_time:
                    s_next = transition(s, u_star, params.Total_vehicles)
                    cont = (
                        (1.0 - params.p_inc) * V[t + 1][k].get(s_next, 0.0)
                        + params.p_inc * V[t + 1][k + 1].get(s_next, 0.0)
                    )
                    stage_value += cont

                V[t][k][s] = float(stage_value)
                u_opt[t][k][s] = int(u_star)

            for s in S[t][k]:
                dV_dl[t][k][s] = finite_diff_dV(V[t][k], s, "l", params.Total_vehicles)
                dV_dr5[t][k][s] = finite_diff_dV(V[t][k], s, "r5", params.Total_vehicles)
                if not math.isfinite(dV_dl[t][k][s]) or not math.isfinite(dV_dr5[t][k][s]):
                    raise ValueError(f"Non-finite derivative at (t={t}, k={k}, s={s})")

    return V, u_opt, dV_dl, dV_dr5
