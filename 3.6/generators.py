from typing import List, Dict
import random

def build_S_t(Num_time: int, demand_states: List[str]) -> List[List[str]]:
    """
    各時刻の状態集合を返す（levels を返す、あるいは簡易コピー）。
    
    Args:
        Num_time (int): 時間の数。
        demand_states (List[str]): 需要状態のリスト。
    
    Returns:
        List[List[str]]: 各時刻の状態集合。
    """
    return [demand_states[:t + 1] for t in range(Num_time)]

def generate_request_sets(Num_time: int, demand_states: List[str], requests_per_state: List[int]) -> Dict[int, Dict[int, List[str]]]:
    """
    Set_of_requests を構築。
    
    Args:
        Num_time (int): 時間の数。
        demand_states (List[str]): 需要状態のリスト。
        requests_per_state (List[int]): 各状態に対するリクエスト数。
    
    Returns:
        Dict[int, Dict[int, List[str]]]: 各時刻と状態に対するリクエスト集合。
    """
    Set_of_requests = {}
    for t in range(Num_time):
        Set_of_requests[t] = {}
        for k in range(len(demand_states[:t + 1])):
            Set_of_requests[t][k] = [f"r_{t}_{k}_{i}" for i in range(max(1, requests_per_state[k]))]
    return Set_of_requests

def generate_average_costs(Num_time: int, n_states: int, Total_vehicles: int, requests_per_state: List[int], base_cost: float, variation: float, seed: int | None) -> Dict[int, Dict[int, Dict[int, List[float]]]]:
    """
    Cost を生成。形状は Cost[t][k][v] -> list[float]。
    
    Args:
        Num_time (int): 時間の数。
        n_states (int): 状態の数。
        Total_vehicles (int): 総車両数。
        requests_per_state (List[int]): 各状態に対するリクエスト数。
        base_cost (float): 基本コスト。
        variation (float): コストの変動幅。
        seed (int | None): 乱数シード。
    
    Returns:
        Dict[int, Dict[int, Dict[int, List[float]]]: 各時刻、状態、車両に対するコストの辞書。
    """
    if seed is not None:
        random.seed(seed)
    
    Cost = {}
    for t in range(Num_time):
        Cost[t] = {}
        for k in range(n_states):
            Cost[t][k] = {}
            for v in range(Total_vehicles):
                cost_variation = random.uniform(-variation * base_cost, variation * base_cost)
                Cost[t][k][v] = [base_cost + cost_variation for _ in range(requests_per_state[k])]
    return Cost

def generate_average_penalties(Num_time: int, n_states: int, requests_per_state: List[int], base_penalty: float, variation: float, seed: int | None) -> Dict[int, Dict[int, List[float]]]:
    """
    Penalty を生成。
    
    Args:
        Num_time (int): 時間の数。
        n_states (int): 状態の数。
        requests_per_state (List[int]): 各状態に対するリクエスト数。
        base_penalty (float): 基本ペナルティ。
        variation (float): ペナルティの変動幅。
        seed (int | None): 乱数シード。
    
    Returns:
        Dict[int, Dict[int, List[float]]]: 各時刻、状態に対するペナルティの辞書。
    """
    if seed is not None:
        random.seed(seed)
    
    Penalty = {}
    for t in range(Num_time):
        Penalty[t] = {}
        for k in range(n_states):
            penalty_variation = random.uniform(-variation * base_penalty, variation * base_penalty)
            Penalty[t][k] = [base_penalty + penalty_variation for _ in range(requests_per_state[k])]
    return Penalty