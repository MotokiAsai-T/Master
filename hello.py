"""
簡易パラメータ生成ツール
- Pythonで実装
- 関数名・リスト名は英語を使用
- コメントは日本語中心
"""

import argparse
import json
import random
from typing import List, Sequence, Dict

# === パラメータ生成関数 ===

def validate_transition_matrix(P: Sequence[Sequence[float]]) -> bool:
    """需要遷移確率行列の検証（各行が1に近いか）"""
    for i, row in enumerate(P):
        s = sum(row)
        if abs(s - 1.0) > 1e-6:
            # 行和が1でない場合は False を返す
            return False
    return True

def generate_random_values(base: float, n: int, variation: float = 0.1, seed: int = None) -> List[float]:
    """
    基準値 base を基に n 個の値を生成する（ランダムに +/- variation 割合で変動）
    variation は比率（例: 0.2 は ±20%）
    """
    if seed is not None:
        random.seed(seed)
    values = []
    for _ in range(n):
        factor = 1.0 + random.uniform(-variation, variation)
        values.append(base * factor)
    return values

def create_request_structure(demand_states: Sequence[str], requests_per_state: Sequence[int]) -> List[Dict]:
    """
    各需要状態ごとのリクエスト数を機械的に扱える構造で返す
    例: [{"state":"low", "requests": 2}, ...]
    """
    if len(demand_states) != len(requests_per_state):
        raise ValueError("demand_states and requests_per_state must have same length")
    structure = []
    for s, r in zip(demand_states, requests_per_state):
        structure.append({"state": s, "requests": int(r)})
    return structure

def sample_demand_sequence(Migration_requires: Sequence[Sequence[float]], start_state: int, length: int, seed: int = None) -> List[int]:
    """
    遷移行列 Migration_requires を用いて需要状態の系列をサンプルする（マルコフ連鎖）
    start_state はインデックス
    """
    if seed is not None:
        random.seed(seed)
    n_states = len(Migration_requires)
    seq = [start_state]
    for _ in range(length - 1):
        cur = seq[-1]
        probs = Migration_requires[cur]
        # 累積和でサンプリング（基本的な手法）
        r = random.random()
        cum = 0.0
        for j, p in enumerate(probs):
            cum += p
            if r <= cum:
                seq.append(j)
                break
    return seq

def build_S_t(Num_time: int, demand_states: Sequence[str]) -> List[List[str]]:
    """各時刻 t の需要状態集合 S_t を返す（ここでは全時刻で同じ状態集合を簡易に返す）"""
    return [list(demand_states) for _ in range(Num_time)]

def generate_request_sets(Num_time: int, demand_states: Sequence[str], requests_per_state: Sequence[int]) -> Dict[int, Dict[int, List[str]]]:
    """
    各時刻 t, 各需要状態 k に対するリクエスト集合 R_{t,k} を生成する。
    返却形式: Set_of_requests[t][k] -> list of request IDs (例: "r_0_1_0")
    """
    Set_of_requests: Dict[int, Dict[int, List[str]]] = {}
    for t in range(Num_time):
        Set_of_requests[t] = {}
        for k in range(len(demand_states)):
            cnt = int(requests_per_state[k])
            Set_of_requests[t][k] = [f"r_{t}_{k}_{i}" for i in range(cnt)]
    return Set_of_requests

def generate_average_costs(Num_time: int, n_states: int, Total_vehicles: int, requests_per_state: Sequence[int],
                           base_cost: float, variation: float = 0.2, seed: int = None) -> Dict[int, Dict[int, Dict[int, List[float]]]]:
    """
    平均コスト Cost_{t,k}^{v,i} を生成する（t: 時刻, k: 需要状態インデックス, v: 車両インデックス, i: リクエストインデックス）
    出力: Cost[t][k][v] -> list of cost for each request i
    """
    if seed is not None:
        random.seed(seed + 100)  # 別シード空間
    Cost: Dict[int, Dict[int, Dict[int, List[float]]]] = {}
    for t in range(Num_time):
        Cost[t] = {}
        for k in range(n_states):
            cnt = int(requests_per_state[k])
            Cost[t][k] = {}
            for v in range(Total_vehicles):
                # 各リクエスト i に対する平均コストを基準値からランダム生成
                Cost[t][k][v] = [base_cost * (1.0 + random.uniform(-variation, variation)) for _ in range(cnt)]
    return Cost

def generate_average_penalties(Num_time: int, n_states: int, requests_per_state: Sequence[int],
                               base_penalty: float, variation: float = 0.2, seed: int = None) -> Dict[int, Dict[int, List[float]]]:
    """
    平均ペナルティ Penalty_{t,k}^i を生成する（t: 時刻, k: 需要状態, i: リクエスト）
    返却形式: Penalty[t][k] -> list of penalties per request i
    """
    if seed is not None:
        random.seed(seed + 200)
    Penalty: Dict[int, Dict[int, List[float]]] = {}
    for t in range(Num_time):
        Penalty[t] = {}
        for k in range(n_states):
            cnt = int(requests_per_state[k])
            Penalty[t][k] = [base_penalty * (1.0 + random.uniform(-variation, variation)) for _ in range(cnt)]
    return Penalty

# === CLI / 実行例 ===

def main():
    parser = argparse.ArgumentParser(description="Generate parameters according to AGENTS.md guidelines")
    parser.add_argument("--p_inc", type=float, default=0.6, help="需要数増加の確率（デフォルト0.6）")
    parser.add_argument("--base_cost", type=float, default=10.0, help="コストの基準値")
    parser.add_argument("--base_penalty", type=float, default=50.0, help="ペナルティの基準値")
    parser.add_argument("--variation", type=float, default=0.2, help="ランダム変動の割合（例:0.2 は ±20%）")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード（再現性用）")
    parser.add_argument("--transition", type=str, default=None, help="需要遷移確率行列(JSON形式). 例: '[[0.8,0.2],[0.1,0.9]]'")
    parser.add_argument("--Num_time", type=int, default=5, help="期間 Num_time（時刻数）")
    parser.add_argument("--Total_vehicles", type=int, default=10, help="総車両数 Total_vehicles")
    parser.add_argument("--theta1", type=float, default=1.0, help="パラメータ theta_1（任意、デフォルト1）")
    parser.add_argument("--theta2", type=float, default=1.0, help="パラメータ theta_2（任意、デフォルト1）")
    args = parser.parse_args()

    # 入力したパラメータをまとめて表示（確認用）
    print("Input parameters:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # 基本設定
    Num_time = args.Num_time
    Total_vehicles = args.Total_vehicles
    theta1 = args.theta1
    theta2 = args.theta2
    p_inc = getattr(args, "p_inc", 0.6)

    # 需要状態を二項分岐モデルに基づき構築
    # time t の状態数 = t+1, グローバルな状態数 n_states = sum_{t=0}^{Num_time-1} (t+1) = Num_time*(Num_time+1)//2
    levels: List[List[str]] = []
    idx_map = {}
    demand_states: List[str] = []
    idx = 0
    for t in range(Num_time):
        level = []
        for j in range(t + 1):
            name = f"state_{t}_{j}"  # (時刻 t, インデックス j) を表すラベル
            level.append(name)
            idx_map[(t, j)] = idx
            demand_states.append(name)
            idx += 1
        levels.append(level)
    n_states = len(demand_states)

    # 各グローバル状態に対するリクエスト数を決定（簡易ルール: レベル内インデックス j に依存）
    requests_per_state = []
    for t in range(Num_time):
        for j in range(t + 1):
            requests_per_state.append(max(1, j + 1))

    # コストとペナルティの生成（state 数に合わせる）
    costs = generate_random_values(args.base_cost, n_states, variation=args.variation, seed=args.seed)
    penalties = generate_random_values(args.base_penalty, n_states, variation=args.variation,
                                       seed=(None if args.seed is None else args.seed + 1))

    # 需要遷移行列 Migration_requires の構築（指定があればそれを使用、無ければ二項分岐に従う行列を作る）
    if args.transition:
        Migration_requires = json.loads(args.transition)
    else:
        # Migration_requires は n_states x n_states 、状態 (t,j) から (t+1,j) と (t+1,j+1) に遷移
        Migration_requires = [[0.0 for _ in range(n_states)] for _ in range(n_states)]
        for t in range(Num_time):
            for j in range(t + 1):
                row = idx_map[(t, j)]
                if t == Num_time - 1:
                    # 末端は自己ループ
                    Migration_requires[row][row] = 1.0
                else:
                    child_low = idx_map[(t + 1, j)]
                    child_high = idx_map[(t + 1, j + 1)]
                    Migration_requires[row][child_low] = 1.0 - p_inc
                    Migration_requires[row][child_high] = p_inc

    if not validate_transition_matrix(Migration_requires):
        raise ValueError("Transition matrix rows must sum to 1. Provided matrix is invalid.")

    # requests_structure はグローバルな demand_states を使って作成
    requests_structure = create_request_structure(demand_states, requests_per_state)

    # サンプル需要列を生成（start は時刻0の唯一の状態）
    start_idx = idx_map[(0, 0)]
    seq = sample_demand_sequence(Migration_requires, start_state=start_idx, length=Num_time, seed=args.seed)

    # S_t（各時刻の需要状態集合）は levels を利用
    Set_demand_state = levels

    # R_{t,k}（各時刻・各需要状態のリクエスト集合）：変数名 Set_of_requests を使用
    Set_of_requests = generate_request_sets(Num_time, demand_states, requests_per_state)

    # 平均コスト Cost と 平均ペナルティ Penalty を生成（n_states を渡す）
    Cost = generate_average_costs(Num_time, n_states, Total_vehicles, requests_per_state,
                               base_cost=args.base_cost, variation=args.variation, seed=args.seed)
    Penalty = generate_average_penalties(Num_time, n_states, requests_per_state,
                                   base_penalty=args.base_penalty, variation=args.variation, seed=args.seed)

    # 要約出力（大きな構造はサンプルを表示）
    print("Set_demand_state:", Set_demand_state)
    print("\nGenerated summary:")
    print("  Num_time:", Num_time, " Total_vehicles:", Total_vehicles)
    print("  theta1:", theta1, " theta2:", theta2)
    print("  demand states:", demand_states)
    print("  sample Set_demand_state[0]:", Set_demand_state[0])
    print("  sample Set_of_requests[0][0]:", Set_of_requests[0][0])
    # Cost と Penalty の存在チェックの上でサンプルを表示
    sample_C = Cost.get(0, {}).get(0, {}).get(0, [None])[0] if isinstance(Cost, dict) else None
    sample_Penalty = Penalty.get(0, {}).get(0, [None])[0] if isinstance(Penalty, dict) else None
    print("  sample Cost[0][0][0][0]:", sample_C)
    print("  sample Penalty[0][0][0]:", sample_Penalty)

if __name__ == "__main__":
    main()