from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ModelParams:
    Num_time: int = 5
    Total_vehicles: int = 20
    p_inc: float = 0.5
    d0: int = 3
    alpha: int = 1

    N_od: int = 3
    rho: list[float] | None = None

    theta_p: float = 1.0
    theta_c: float = 1.0
    eta: float = 1.5
    L0: float = 1.0
    eps_conv: float = 1e-6
    max_iter: int = 5000

    base_cost_proc: float = 3.0
    base_cost_wait: float = 1.5
    base_selftravel: float = 8.0
    variation: float = 0.3
    seed: int | None = None

    C0: Dict[int, Dict[int, float]] | None = None
    C_proc: Dict[int, Dict[int, np.ndarray]] | None = None
    S_od: Dict[int, Dict[int, np.ndarray]] | None = None
    RealCost: Dict[int, Dict[int, np.ndarray]] | None = None
    RealSelfTravel: Dict[int, Dict[int, np.ndarray]] | None = None
    RealWaitCost: Dict[int, Dict[int, np.ndarray]] | None = None


def validate_input_params(params: ModelParams) -> None:
    if params.Total_vehicles <= 0:
        raise ValueError("Total_vehicles must be > 0")
    if not (0.0 <= params.p_inc <= 1.0):
        raise ValueError("p_inc must be in [0, 1]")
    if params.d0 < 0 or params.alpha < 0:
        raise ValueError("d0 and alpha must be >= 0")
    if params.N_od < 1:
        raise ValueError("N_od must be >= 1")

    if params.rho is None:
        params.rho = [1.0 / params.N_od] * params.N_od

    if len(params.rho) != params.N_od:
        raise ValueError("len(rho) must equal N_od")
    if abs(sum(params.rho) - 1.0) >= 1e-9:
        raise ValueError("sum(rho) must be 1.0")

    if params.eta <= 1.0:
        raise ValueError("eta must be > 1")
    if params.L0 <= 0.0:
        raise ValueError("L0 must be > 0")
    if params.eps_conv <= 0.0:
        raise ValueError("eps_conv must be > 0")


def calc_demand(t: int, k: int, d0: int, alpha: int) -> int:
    return int(d0 + alpha * k)


def calc_demand_od(t: int, k: int, d0: int, alpha: int, rho: list[float]) -> list[int]:
    d_total = calc_demand(t, k, d0, alpha)
    n_od = len(rho)

    raw = [rho[n] * d_total for n in range(n_od)]
    base = [math.floor(x) for x in raw]
    remainder = d_total - sum(base)
    frac = [raw[n] - base[n] for n in range(n_od)]

    order = sorted(range(n_od), key=lambda n: frac[n], reverse=True)
    for n in order[:remainder]:
        base[n] += 1

    return [int(x) for x in base]


def generate_cost_data(
    params: ModelParams,
) -> tuple[
    Dict[int, Dict[int, float]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
    Dict[int, Dict[int, np.ndarray]],
]:
    validate_input_params(params)
    rng = np.random.default_rng(params.seed)

    d_max = params.d0 + params.alpha * params.Num_time
    n_od = params.N_od
    l = params.Total_vehicles

    c0: Dict[int, Dict[int, float]] = {}
    c_proc: Dict[int, Dict[int, np.ndarray]] = {}
    s_od: Dict[int, Dict[int, np.ndarray]] = {}
    real_cost: Dict[int, Dict[int, np.ndarray]] = {}
    real_self: Dict[int, Dict[int, np.ndarray]] = {}
    real_wait: Dict[int, Dict[int, np.ndarray]] = {}

    denom = max(1, n_od - 1)
    od_scale = np.array([0.8 + 0.4 * n / denom for n in range(n_od)], dtype=float)

    for t in range(params.Num_time + 1):
        c0[t] = {}
        c_proc[t] = {}
        s_od[t] = {}
        real_cost[t] = {}
        real_self[t] = {}
        real_wait[t] = {}
        for k in range(t + 1):
            c0_val = float(params.base_cost_wait + params.variation * rng.standard_normal())
            c0_val = max(0.1, c0_val)
            c0[t][k] = c0_val

            c_proc_vec = params.base_cost_proc * od_scale + params.variation * rng.standard_normal(n_od)
            c_proc_vec = np.maximum(0.1, c_proc_vec).astype(float)
            c_proc[t][k] = c_proc_vec

            s_od_vec = params.base_selftravel * od_scale + params.variation * rng.standard_normal(n_od)
            s_od_vec = np.maximum(s_od_vec, c_proc_vec + 1.0).astype(float)
            s_od[t][k] = s_od_vec

            rc = np.empty((l, d_max, n_od), dtype=float)
            for v in range(l):
                for i in range(d_max):
                    for n in range(n_od):
                        rc[v, i, n] = c_proc_vec[n] + params.variation * rng.standard_normal()
            rc = np.maximum(0.01, rc)
            real_cost[t][k] = rc

            rs = np.empty((d_max, n_od), dtype=float)
            for i in range(d_max):
                for n in range(n_od):
                    rs[i, n] = s_od_vec[n] + params.variation * rng.standard_normal()
            rs = np.maximum(0.01, rs)
            real_self[t][k] = rs

            rw = c0_val + params.variation * rng.standard_normal(l)
            rw = np.maximum(0.01, rw).astype(float)
            real_wait[t][k] = rw

    return c0, c_proc, s_od, real_cost, real_self, real_wait
