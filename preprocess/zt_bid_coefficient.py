import math

import pandas as pd


DEFAULT_M = 2.0
EPSILON = 1e-12


def calculate_zt_bid_coefficient(
        time_step_data,
        period_data=None,
        zt_bgtleft=None,
        m=DEFAULT_M,
        total_cost='leastWinningBid'):
    period_data = period_data if period_data is not None else time_step_data
    if any(field not in period_data.columns for field in ['pValue', 'bid', total_cost]):
        return 0.0

    total_value = period_data['pValue'].sum()
    total_bid = period_data['bid'].sum()
    if total_value <= 0 or total_bid <= 0 or zt_bgtleft is None or zt_bgtleft <= 0 or m <= 1:
        return 0.0

    lambda1 = total_value / total_bid
    candidates = period_data[['pValue', 'bid', total_cost]].copy()
    # candidates = candidates.apply(pd.to_numeric, errors='coerce').dropna()
    candidates = candidates[
        (candidates['pValue'] > 0)
        & (candidates['bid'] > 0)
        & (candidates[total_cost] > candidates['bid']) # 当前 bid 不够高，只有追加出价之后才可能赢
    ]
    if candidates.empty:
        return 0.0

    candidates['threshold'] = candidates['pValue'] / (
        m * (candidates[total_cost] - candidates['bid'])
    )
    candidates = candidates[candidates['threshold'] > 0].sort_values('threshold', ascending=False)
    if candidates.empty:
        return 0.0

    lambda2 = None
    cumulative_cost = 0.0
    thresholds = candidates['threshold'].tolist()
    costs = candidates[total_cost].tolist()
    max_lambda2 = lambda1 * (m - 1) / m

    for index, cost in enumerate(costs):
        cumulative_cost += cost
        lower_threshold = thresholds[index + 1] if index + 1 < len(thresholds) else 0.0
        upper_threshold = thresholds[index]
        discriminant = (m - 1) ** 2 - 4 * m * (cumulative_cost / zt_bgtleft - 1)
        if discriminant < -EPSILON:
            continue

        sqrt_discriminant = math.sqrt(max(discriminant, 0.0))
        for sign in (-1, 1):
            candidate_lambda2 = lambda1 * ((m - 1) + sign * sqrt_discriminant) / (2 * m)
            if (
                    candidate_lambda2 > EPSILON
                    and candidate_lambda2 < max_lambda2 - EPSILON
                    and candidate_lambda2 <= upper_threshold + EPSILON
                    and candidate_lambda2 + EPSILON >= lower_threshold):
                lambda2 = candidate_lambda2
                break
        if lambda2 is not None:
            break

    if lambda2 is None:
        return 0.0

    return lambda2
