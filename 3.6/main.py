# filepath: /binary_model_1/main.py

import json
from typing import List, Dict, Tuple
from generators import generate_request_sets, generate_average_costs, generate_average_penalties
from validators import validate_transition_matrix
import cli

def main():
    # コマンドライン引数を処理
    args = cli.parse_arguments()

    # 必要なパラメータを取得
    Num_time = args.Num_time
    Total_vehicles = args.Total_vehicles
    p_inc = args.p_inc
    base_cost = args.base_cost
    base_penalty = args.base_penalty
    variation = args.variation
    seed = args.seed
    transition_matrix = args.transition
    theta1 = args.theta1
    theta2 = args.theta2
    start_state = args.start_state

    # 遷移行列の検証
    if transition_matrix:
        P = json.loads(transition_matrix)
        if not validate_transition_matrix(P):
            raise ValueError("遷移行列の検証に失敗しました。")

    # 需要状態の生成
    demand_states = build_demand_states(Num_time)

    # リクエスト集合の生成
    requests_per_state = [max(1, j + 1) for j in range(Num_time)]
    Set_of_requests = generate_request_sets(Num_time, demand_states, requests_per_state)

    # コストとペナルティの生成
    n_states = Num_time * (Num_time + 1) // 2
    Cost = generate_average_costs(Num_time, n_states, Total_vehicles, requests_per_state, base_cost, variation, seed)
    Penalty = generate_average_penalties(Num_time, n_states, requests_per_state, base_penalty, variation, seed)

    # Step0の初期状態を設定
    state_DP = initialize_dp_state(Total_vehicles)

    # 結果を表示
    print_initial_state(args, state_DP)

def build_demand_states(Num_time: int) -> List[str]:
    # 需要状態を構築する
    demand_states = []
    for t in range(Num_time):
        for j in range(t + 1):
            demand_states.append(f'state_{t}_{j}')
    return demand_states

def initialize_dp_state(Total_vehicles: int) -> Tuple[int, int, int, int]:
    # DPの初期状態を設定
    return (Total_vehicles, 0, 0, 0)

def print_initial_state(args, state_DP):
    # 初期状態を表示
    print(f"Input parameters: {vars(args)}")
    print(f"Step0: initial state set to '{args.start_state}'")
    print(f"Step0: DP initial state_DP: {state_DP}")

if __name__ == "__main__":
    main()