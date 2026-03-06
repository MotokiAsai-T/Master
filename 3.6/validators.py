def validate_transition_matrix(P: list[list[float]]) -> bool:
    """
    各行の和が 1 に近いか検証する関数。

    Args:
        P (list[list[float]]): 遷移行列。

    Returns:
        bool: 各行の和が 1 に近い場合は True、そうでない場合は False。
    """
    for row in P:
        if not (0.99 <= sum(row) <= 1.01):
            return False
    return True


def validate_requests_per_state(requests_per_state: list[int], n_states: int) -> None:
    """
    requests_per_state の長さが n_states と一致するか検証する関数。

    Args:
        requests_per_state (list[int]): 各状態に対するリクエスト数のリスト。
        n_states (int): グローバル状態数。

    Raises:
        ValueError: requests_per_state の長さが n_states と一致しない場合。
    """
    if len(requests_per_state) != n_states:
        raise ValueError("requests_per_state の長さは n_states と一致する必要があります。") 


def validate_start_state(start_state: str, demand_states: list[str]) -> None:
    """
    start_state が demand_states に存在するか検証する関数。

    Args:
        start_state (str): 開始状態のラベル。
        demand_states (list[str]): 需要状態のリスト。

    Raises:
        ValueError: start_state が demand_states に存在しない場合。
    """
    if start_state not in demand_states:
        raise ValueError(f"start_state '{start_state}' は demand_states に存在しません。")