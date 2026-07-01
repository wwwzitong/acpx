def calculate_zt_bid_coefficient(time_step_data):
    total_value = time_step_data['pValue'].sum()
    return time_step_data['bid'].sum() / total_value if total_value > 0 else 0
