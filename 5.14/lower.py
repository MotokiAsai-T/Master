from __future__ import annotations

from typing import Dict

import numpy as np

from model import ModelParams, calc_demand_od, validate_input_params


def greedy_match(
    t: int,
    k: int,
    m_star_tk: np.ndarray,
    params: ModelParams,
) -> tuple[Dict[tuple[int, int, int], bool], Dict[tuple[int, int], bool], Dict[int, bool]]:
    if params.RealCost is None or params.RealSelfTravel is None or params.RealWaitCost is None:
        raise ValueError("RealCost/RealSelfTravel/RealWaitCost must be provided")

    d_od_n = calc_demand_od(t, k, params.d0, params.alpha, params.rho or [])
    m_clipped = np.asarray(m_star_tk, dtype=int).copy()

    for n in range(params.N_od):
        m_clipped[n] = max(0, min(int(m_clipped[n]), int(d_od_n[n])))

    assert int(np.sum(m_clipped)) <= params.Total_vehicles

    edges: list[tuple[float, int, int, int]] = []
    for v in range(params.Total_vehicles):
        for n in range(params.N_od):
            for i in range(d_od_n[n]):
                c = float(params.RealCost[t][k][v, i, n])
                edges.append((c, v, i, n))
    edges.sort(key=lambda row: row[0])

    x: Dict[tuple[int, int, int], bool] = {}
    y: Dict[tuple[int, int], bool] = {}
    wait: Dict[int, bool] = {}

    free_vehicles = set(range(params.Total_vehicles))
    assigned_requests: set[tuple[int, int]] = set()
    count_n = np.zeros(params.N_od, dtype=int)

    for _c, v, i, n in edges:
        if v not in free_vehicles:
            continue
        if (i, n) in assigned_requests:
            continue
        if count_n[n] >= m_clipped[n]:
            continue

        x[(v, i, n)] = True
        free_vehicles.remove(v)
        assigned_requests.add((i, n))
        count_n[n] += 1

        if np.all(count_n == m_clipped):
            break

    for n in range(params.N_od):
        for i in range(d_od_n[n]):
            if (i, n) not in assigned_requests:
                y[(i, n)] = True

    for v in free_vehicles:
        wait[v] = True

    return x, y, wait


def simulate_phase2(
    v_star: Dict[int, Dict[int, np.ndarray]],
    m_star: Dict[int, Dict[int, np.ndarray]],
    params: ModelParams,
) -> list[dict]:
    validate_input_params(params)
    if params.RealCost is None or params.RealSelfTravel is None or params.RealWaitCost is None:
        raise ValueError("RealCost/RealSelfTravel/RealWaitCost must be provided")

    rng = np.random.default_rng(params.seed)
    k = 0
    results: list[dict] = []

    for t in range(params.Num_time + 1):
        m_policy = np.asarray(m_star[t][k], dtype=int)
        x, y, wait = greedy_match(t, k, m_policy, params)

        d_od_n = calc_demand_od(t, k, params.d0, params.alpha, params.rho or [])
        assigned_per_od = np.zeros(params.N_od, dtype=int)
        selftrav_per_od = np.zeros(params.N_od, dtype=int)

        assign_cost = 0.0
        for (v, i, n), flag in x.items():
            if flag:
                assigned_per_od[n] += 1
                assign_cost += float(params.RealCost[t][k][v, i, n])

        selftrav_cost = 0.0
        for (i, n), flag in y.items():
            if flag:
                selftrav_per_od[n] += 1
                selftrav_cost += float(params.RealSelfTravel[t][k][i, n])

        wait_cost = 0.0
        for v, flag in wait.items():
            if flag:
                wait_cost += float(params.RealWaitCost[t][k][v])

        total_cost = assign_cost + wait_cost + selftrav_cost

        results.append(
            {
                "t": int(t),
                "k": int(k),
                "v_star": np.asarray(v_star[t][k], dtype=float),
                "m_policy": m_policy.copy(),
                "m_star": assigned_per_od.copy(),
                "demand_per_od": list(map(int, d_od_n)),
                "assigned_per_od": assigned_per_od.copy(),
                "selftrav_per_od": selftrav_per_od.copy(),
                "waited_vehicles": int(len(wait)),
                "x": x,
                "y": y,
                "assign_cost": float(assign_cost),
                "wait_cost": float(wait_cost),
                "selftrav_cost": float(selftrav_cost),
                "total_cost": float(total_cost),
            }
        )

        if t < params.Num_time and rng.random() < params.p_inc:
            k += 1

    return results
