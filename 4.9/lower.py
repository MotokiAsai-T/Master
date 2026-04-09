from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from model import ModelParams, validate_input_params
from utils import calc_D, get_l, transition

State = tuple[int, int, int, int, int]


def greedy_match(
    t: int,
    k: int,
    s: State,
    u_opt: Dict[int, Dict[int, Dict[State, int]]],
    params: ModelParams,
) -> Tuple[Dict[Tuple[int, int], int], Dict[int, int]]:
    """貪欲法で車両-リクエスト割当 x と拒否 y を返す。"""
    if params.RealCost is None or params.RealPenalty is None:
        raise ValueError("RealCost and RealPenalty must be provided")

    m_star = int(u_opt.get(t, {}).get(k, {}).get(s, 0))
    l = get_l(s, params.Total_vehicles)
    req_n = max(0, calc_D(t, k, params.d0, params.alpha))
    m_star = min(m_star, l, req_n)

    edges: List[tuple[float, int, int]] = []
    rc = params.RealCost.get(t, {}).get(k, [])
    for v in range(min(l, len(rc))):
        for i in range(min(req_n, len(rc[v]))):
            edges.append((float(rc[v][i]), v, i))
    edges.sort(key=lambda x: x[0])

    x: Dict[Tuple[int, int], int] = {}
    y: Dict[int, int] = {}

    used_v = set()
    used_i = set()

    for _, v, i in edges:
        if len(x) >= m_star:
            break
        if v in used_v or i in used_i:
            continue
        used_v.add(v)
        used_i.add(i)
        x[(v, i)] = 1

    for i in range(req_n):
        if i not in used_i:
            y[i] = 1

    return x, y


def simulate_phase2(
    u_opt: Dict[int, Dict[int, Dict[State, int]]],
    params: ModelParams,
    seed: int | None = None,
) -> List[Dict]:
    """実現需要パスに対して Phase 2 を実行し時系列結果を返す。"""
    validate_input_params(params)
    if params.RealCost is None or params.RealPenalty is None:
        raise ValueError("RealCost and RealPenalty must be provided")

    rng = np.random.default_rng(seed)

    results: List[Dict] = []
    s: State = (0, 0, 0, 0, 0)
    k = 0

    for t in range(params.Num_time + 1):
        x, y = greedy_match(t, k, s, u_opt, params)
        m_policy = int(u_opt.get(t, {}).get(k, {}).get(s, 0))

        # 方策値と実行値を分離して保持する（実行値は制約下での実際の処理件数）。
        l = get_l(s, params.Total_vehicles)
        req_n = max(0, calc_D(t, k, params.d0, params.alpha))
        m_star = min(max(m_policy, 0), l, req_n)

        assign_cost = 0.0
        for (v, i), _flag in x.items():
            assign_cost += float(params.RealCost[t][k][v][i])

        reject_cost = 0.0
        for i, _flag in y.items():
            reject_cost += float(params.RealPenalty[t][k][i])

        total_cost = assign_cost + reject_cost

        results.append(
            {
                "t": t,
                "k": k,
                "s": s,
                "x": x,
                "y": y,
                "available": l,
                "demand": req_n,
                "m_policy": m_policy,
                "m_star": m_star,
                "assigned": len(x),
                "rejected": len(y),
                "assign_cost": assign_cost,
                "reject_cost": reject_cost,
                "total_cost": total_cost,
            }
        )

        if t == params.Num_time:
            break

        s = transition(s, m_star, params.Total_vehicles)
        if rng.uniform(0.0, 1.0) < params.p_inc:
            k += 1

    return results
