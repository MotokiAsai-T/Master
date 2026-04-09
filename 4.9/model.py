from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

State = tuple[int, int, int, int, int]
NestedFloat = Dict[int, Dict[int, List[float]]]
RealCostType = Dict[int, Dict[int, List[List[float]]]]


@dataclass
class ModelParams:
    """モデル全体で使うパラメータを保持する。"""

    Num_time: int = 5
    Total_vehicles: int = 20
    p_inc: float = 0.5
    d0: int = 3
    alpha: int = 1

    Cost: NestedFloat | None = None
    Penalty: NestedFloat | None = None
    RealCost: RealCostType | None = None
    RealPenalty: NestedFloat | None = None

    theta1: float = 1.0
    theta2: float = 1.0
    alpha_lr: float = 0.01
    eps: float = 1e-4
    eps_bisec: float = 1e-8

    base_cost: float = 3.0
    base_penalty: float = 10.0
    variation: float = 0.2
    seed: int | None = None


def calc_demand(t: int, k: int, d0: int, alpha: int) -> int:
    """需要量 D(t, k) を返す。"""
    return int(d0 + alpha * k)


def validate_input_params(params: ModelParams) -> None:
    """仕様に沿って入力パラメータを検証する。"""
    if params.Total_vehicles <= 0:
        raise ValueError("Total_vehicles must be > 0")
    if not (0.0 <= params.p_inc <= 1.0):
        raise ValueError("p_inc must be in [0, 1]")
    if params.d0 < 0 or params.alpha < 0:
        raise ValueError("d0 and alpha must be >= 0")


def generate_cost_data(params: ModelParams) -> tuple[NestedFloat, NestedFloat, RealCostType, NestedFloat]:
    """仕様に沿って Cost/Penalty/RealCost/RealPenalty を生成する。"""
    validate_input_params(params)
    rng = np.random.default_rng(params.seed)

    cost: NestedFloat = {}
    penalty: NestedFloat = {}
    real_cost: RealCostType = {}
    real_penalty: NestedFloat = {}

    for t in range(params.Num_time + 1):
        cost[t] = {}
        penalty[t] = {}
        real_cost[t] = {}
        real_penalty[t] = {}

        for k in range(t + 1):
            req_n = max(0, calc_demand(t, k, params.d0, params.alpha))
            c_vec: List[float] = []
            b_vec: List[float] = []

            for _ in range(req_n):
                c = params.base_cost * (1.0 + params.variation * rng.uniform(-1.0, 1.0))
                b = params.base_penalty * (1.0 + params.variation * rng.uniform(-1.0, 1.0))
                # ペナルティはコストより大きくなるように補正
                b = max(b, c + 1e-6)
                c_vec.append(float(c))
                b_vec.append(float(b))

            cost[t][k] = c_vec
            penalty[t][k] = b_vec
            real_penalty[t][k] = list(b_vec)

            v_costs: List[List[float]] = []
            for _v in range(params.Total_vehicles):
                row = []
                for i in range(req_n):
                    rc = c_vec[i] * (1.0 + params.variation * rng.uniform(-1.0, 1.0))
                    row.append(float(max(0.0, rc)))
                v_costs.append(row)
            real_cost[t][k] = v_costs

    return cost, penalty, real_cost, real_penalty
