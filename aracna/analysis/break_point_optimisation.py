import numpy as np


def get_break_points(arr, sum_dim=True):
    if sum_dim:
        _arr = arr[~np.isnan(arr).any(axis=1)]
        return int((_arr[:-1] != _arr[1:]).any(axis=1).sum())
    return int((arr[:-1] != arr[1:]).sum())


def find_optimal_sequence(p, lambd):
    N, K = p.shape
    V = np.zeros((N, K))
    path = np.zeros((N, K), dtype=int)
    vec1 = np.arange(K).reshape(1, -1)
    vec2 = np.arange(K).reshape(-1, 1)
    multiplier = vec1 != vec2

    # Initialization
    V[0, :] = -np.log(p[0, :])

    for i in range(1, N):
        costs = V[i - 1, :].reshape(-1, 1) + lambd * multiplier

        # Vectorized minimum cost and argmin
        min_costs = np.min(costs, axis=0)
        min_states = np.argmin(costs, axis=0)

        V[i, :] = min_costs - np.log(p[i, :])
        path[i, :] = min_states

    # Backtrack to find optimal path
    optimal_sequence = np.zeros(N, dtype=int)
    optimal_sequence[N - 1] = np.argmin(V[N - 1, :])

    for i in range(N - 2, -1, -1):
        optimal_sequence[i] = path[i + 1, optimal_sequence[i + 1]]

    return optimal_sequence


def opt_num_break_pts(prob_arr, lambda_value):
    max_tot_cn = 8
    cat_ls = [(i - j, j) for i in range(max_tot_cn + 1) for j in range(i // 2 + 1)]
    optimal_sequence = find_optimal_sequence(prob_arr, lambda_value)
    new_seq = np.array(cat_ls)[optimal_sequence]
    num_break_pts = get_break_points(new_seq)
    return new_seq, num_break_pts


def bisection_method_to_match_break_pts(
    prob_arr,
    target_break_pts,
    opt_func=opt_num_break_pts,
    tolerance=20,
    max_iter=10,
    lambda_min=10,
    lambda_max=10000,
):
    for _ in range(max_iter):
        # Midpoint for bisection method
        lambda_value = (lambda_min + lambda_max) / 2.0

        new_seq, num_break_pts = opt_func(prob_arr, lambda_value)
        # Calculate the difference from the target
        diff = num_break_pts - target_break_pts

        # Check if the difference is within the tolerance
        if abs(diff) < tolerance:
            return lambda_value, new_seq

        # Adjust the bisection bounds
        if diff < 0:
            lambda_max = lambda_value  # Too many break points, decrease lambda/window
        else:
            lambda_min = lambda_value  # Too few break points, increase lambda/window

    # If max_iter is reached, return the best estimate
    return lambda_value, new_seq
