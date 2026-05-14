from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np

from lower import simulate_phase2
from model import ModelParams, calc_demand, calc_demand_od, generate_cost_data, validate_input_params
from upper import solve_upper
from utils import calc_MC


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-OD pricing + assignment model")

    parser.add_argument("--Num_time", type=int, default=5)
    parser.add_argument("--Total_vehicles", type=int, default=20)
    parser.add_argument("--p_inc", type=float, default=0.5)
    parser.add_argument("--d0", type=int, default=3)
    parser.add_argument("--alpha", type=int, default=1)
    parser.add_argument("--N_od", type=int, default=3)
    parser.add_argument("--rho", nargs="+", type=float, default=None)

    parser.add_argument("--theta_p", type=float, default=1.0)
    parser.add_argument("--theta_c", type=float, default=1.0)
    parser.add_argument("--eta", type=float, default=1.5)
    parser.add_argument("--L0", type=float, default=1.0)
    parser.add_argument("--eps_conv", type=float, default=1e-6)
    parser.add_argument("--max_iter", type=int, default=5000)

    parser.add_argument("--base_cost_proc", type=float, default=3.0)
    parser.add_argument("--base_cost_wait", type=float, default=1.5)
    parser.add_argument("--base_selftravel", type=float, default=8.0)
    parser.add_argument("--variation", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=None)

    return parser


def _print_input_summary(params: ModelParams) -> None:
    print("=== Input Parameters ===")
    print(f"Num_time={params.Num_time}")
    print(f"Total_vehicles={params.Total_vehicles}")
    print(f"p_inc={params.p_inc}")
    print(f"d0={params.d0}")
    print(f"alpha={params.alpha}")
    print(f"N_od={params.N_od}")
    print(f"rho={params.rho}")
    print(f"theta_p={params.theta_p}")
    print(f"theta_c={params.theta_c}")
    print(f"eta={params.eta}")
    print(f"L0={params.L0}")
    print(f"eps_conv={params.eps_conv}")
    print(f"max_iter={params.max_iter}")
    print(f"base_cost_proc={params.base_cost_proc}")
    print(f"base_cost_wait={params.base_cost_wait}")
    print(f"base_selftravel={params.base_selftravel}")
    print(f"variation={params.variation}")
    print(f"seed={params.seed}")


def main() -> None:
    total_start = perf_counter()
    args = build_parser().parse_args()

    params = ModelParams(
        Num_time=args.Num_time,
        Total_vehicles=args.Total_vehicles,
        p_inc=args.p_inc,
        d0=args.d0,
        alpha=args.alpha,
        N_od=args.N_od,
        rho=args.rho,
        theta_p=args.theta_p,
        theta_c=args.theta_c,
        eta=args.eta,
        L0=args.L0,
        eps_conv=args.eps_conv,
        max_iter=args.max_iter,
        base_cost_proc=args.base_cost_proc,
        base_cost_wait=args.base_cost_wait,
        base_selftravel=args.base_selftravel,
        variation=args.variation,
        seed=args.seed,
    )

    validate_input_params(params)

    params.C0, params.C_proc, params.S_od, params.RealCost, params.RealSelfTravel, params.RealWaitCost = generate_cost_data(params)

    _print_input_summary(params)
    total_nodes = (params.Num_time + 1) * (params.Num_time + 2) // 2
    print(f"Total nodes: {total_nodes}")

    d00 = calc_demand(0, 0, params.d0, params.alpha)
    d_od_00 = calc_demand_od(0, 0, params.d0, params.alpha, params.rho or [])
    print(f"D(0,0) = {d00}, D_od = {d_od_00}")

    print("=== Phase 1: Solving [SO-D] for all nodes (FISTA) ===")
    phase1_start = perf_counter()
    v_star, m_star = solve_upper(params)
    phase1_elapsed = perf_counter() - phase1_start

    converged = 0
    for t in range(params.Num_time + 1):
        for k in range(t + 1):
            d_od_n = calc_demand_od(t, k, params.d0, params.alpha, params.rho or [])
            _, grad = calc_MC(
                v_star[t][k],
                d_od_n,
                params.C0[t][k],
                params.C_proc[t][k],
                params.S_od[t][k],
                params.Total_vehicles,
                params.theta_p,
                params.theta_c,
            )
            if np.linalg.norm(grad) < params.eps_conv:
                converged += 1

    print("Phase 1 complete.")
    print(f"Converged nodes: {converged} / {total_nodes}")
    print(f"v_star[0][0] = {v_star[0][0]}")
    print(f"m_star[0][0] = {m_star[0][0]}")
    print(f"Phase 1 elapsed: {phase1_elapsed:.3f} s")

    print("=== Phase 2: Greedy Matching along Realized Path ===")
    phase2_start = perf_counter()
    results = simulate_phase2(v_star, m_star, params)
    phase2_elapsed = perf_counter() - phase2_start

    total_cost = 0.0
    for row in results:
        total_cost += float(row["total_cost"])
        print(
            f"t={row['t']}, k={row['k']} | demand_per_od={row['demand_per_od']} | "
            f"m_star={row['m_star'].tolist()} | assigned_per_od={row['assigned_per_od'].tolist()}"
        )
        print(
            f"  assign_cost={row['assign_cost']:.4f}, wait_cost={row['wait_cost']:.4f}, "
            f"selftrav_cost={row['selftrav_cost']:.4f}, total={row['total_cost']:.4f}"
        )

    print(f"Total cost (Phase 2): {total_cost:.4f}")
    print(f"Phase 2 elapsed: {phase2_elapsed:.3f} s")

    total_elapsed = perf_counter() - total_start
    print(f"Total elapsed: {total_elapsed:.3f} s")


if __name__ == "__main__":
    main()
