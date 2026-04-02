from __future__ import annotations

import argparse
from typing import Dict

from lower import simulate_phase2
from model import ModelParams, generate_cost_data, validate_input_params
from upper import forward_pass, solve_upper
from utils import get_l

State = tuple[int, int, int, int, int]


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数を定義する。"""
    parser = argparse.ArgumentParser(description="需要二項分岐型モデルの最適化")
    parser.add_argument("--Num_time", type=int, default=5)
    parser.add_argument("--Total_vehicles", type=int, default=20)
    parser.add_argument("--p_inc", type=float, default=0.5)
    parser.add_argument("--d0", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=1)

    parser.add_argument("--base_cost", type=float, default=3.0)
    parser.add_argument("--base_penalty", type=float, default=10.0)
    parser.add_argument("--variation", type=float, default=0.2)

    parser.add_argument("--theta1", type=float, default=1.0)
    parser.add_argument("--theta2", type=float, default=1.0)
    parser.add_argument("--alpha_lr", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-4)
    parser.add_argument("--eps_bisec", type=float, default=1e-8)

    parser.add_argument("--seed", type=int, default=None)
    return parser


def print_step0_summary(params: ModelParams, S: Dict[int, Dict[int, set[State]]]) -> None:
    """Step0 の要約を表示する。"""
    print(
        "Input parameters: "
        f"Num_time={params.Num_time}, "
        f"Total_vehicles={params.Total_vehicles}, "
        f"p_inc={params.p_inc}, "
        f"d0={params.d0}, "
        f"alpha={params.alpha}, "
        f"theta1={params.theta1}, "
        f"theta2={params.theta2}, "
        f"alpha_lr={params.alpha_lr}, "
        f"eps={params.eps}, "
        f"eps_bisec={params.eps_bisec}, "
        f"seed={params.seed}"
    )

    s0: State = (0, 0, 0, 0, 0)
    print(f"Step0: DP initial state: {s0}, l={get_l(s0, params.Total_vehicles)}")

    chunks = []
    for t in range(params.Num_time + 1):
        for k in range(t + 1):
            chunks.append(f"S[{t}][{k}]={len(S[t][k])}")
    print("Reachable states: " + ", ".join(chunks))


def main() -> None:
    """エントリーポイント。Step0/Phase1/Phase2 を順に実行する。"""
    parser = build_parser()
    args = parser.parse_args()

    params = ModelParams(
        Num_time=args.Num_time,
        Total_vehicles=args.Total_vehicles,
        p_inc=args.p_inc,
        d0=args.d0,
        alpha=args.alpha,
        theta1=args.theta1,
        theta2=args.theta2,
        alpha_lr=args.alpha_lr,
        eps=args.eps,
        eps_bisec=args.eps_bisec,
        base_cost=args.base_cost,
        base_penalty=args.base_penalty,
        variation=args.variation,
        seed=args.seed,
    )
    validate_input_params(params)

    # Step0: 入力確認とデータ生成
    cost, penalty, real_cost, real_penalty = generate_cost_data(params)
    params.Cost = cost
    params.Penalty = penalty
    params.RealCost = real_cost
    params.RealPenalty = real_penalty

    S = forward_pass(params.Num_time, params.Total_vehicles, params.d0, params.alpha)
    print_step0_summary(params, S)

    # Phase1: 上位問題
    V, u_opt, _dV_dl, _dV_dr5 = solve_upper(params)
    s0: State = (0, 0, 0, 0, 0)

    print("Phase 1 complete.")
    print(f"V[0][0][(0, 0, 0, 0, 0)] = {V[0][0][s0]:.4f}")
    print(f"u_opt[0][0][(0, 0, 0, 0, 0)] = {u_opt[0][0][s0]}")

    # Phase2: 下位問題
    results = simulate_phase2(u_opt, params, seed=params.seed)
    print("Phase 2 simulation result:")

    total = 0.0
    for row in results:
        assigned = len(row["x"])
        rejected = len(row["y"])
        total += float(row["total_cost"])
        print(
            f"  t={row['t']}, k={row['k']}, s={row['s']}, "
            f"m*={row['m_star']}, assigned={assigned}, rejected={rejected}"
        )

    print(f"Total cost: {total:.4f}")


if __name__ == "__main__":
    main()
