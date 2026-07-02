import random
import pandas as pd
from zt_bid_coefficient import DEFAULT_M, calculate_zt_bid_coefficient

DEFAULT_ZT_CONVERSION_RATE = 0.010726

def replay_single_advertiser_zt(
        df,
        target_advertiser,
        m=DEFAULT_M,
        zt_conversion_rate=DEFAULT_ZT_CONVERSION_RATE,
        random_seed=None):
    random_generator = random.Random(random_seed)
    replay_df = df.copy(deep=True)
    replay_df = replay_df.drop(columns=['cost'], errors='ignore')
    replay_df['ztAdvertiserNumber'] = target_advertiser
    replay_df['ztBid'] = 0.0
    replay_df['ztBidCoefficient'] = 0.0
    replay_df['ztBudget'] = 0.0
    replay_df['mtBid'] = replay_df['bid']
    replay_df['originalBid'] = replay_df['bid']
    replay_df['originalIsExposed'] = replay_df['isExposed']
    replay_df['ztRemainingBudget'] = 0.0
    replay_df['mt_cost'] = 0.0
    replay_df['zt_cost'] = 0.0
    replay_df['total_cost'] = 0.0

    auction_keys = ['deliveryPeriodIndex', 'timeStepIndex', 'pvIndex']
    replay_df = _add_least_winning_bid(replay_df, auction_keys)

    target_mask = replay_df['advertiserNumber'] == target_advertiser
    target_data = replay_df[target_mask]
    if target_data.empty:
        return replay_df

    mt_bgtleft = target_data['budget'].iloc[0]
    zt_bgtleft = mt_bgtleft
    replay_df.loc[target_mask, 'ztBudget'] = zt_bgtleft

    for time_step in sorted(target_data['timeStepIndex'].unique()):
        step_target_mask = target_mask & replay_df['timeStepIndex'].eq(time_step)
        step_target_index = replay_df.index[step_target_mask]
        if step_target_index.empty:
            continue

        replay_df.loc[step_target_index, 'remainingBudget'] = mt_bgtleft
        replay_df.loc[step_target_index, 'ztRemainingBudget'] = zt_bgtleft

        step_target_data = replay_df.loc[step_target_index].copy()
        mt_lambda1 = _calculate_lambda1(step_target_data, mt_bgtleft)
        if mt_lambda1 <= 0:
            mt_bid = pd.Series(0.0, index=step_target_index)
        else:
            mt_bid = step_target_data['pValue'] / mt_lambda1

        replay_df.loc[step_target_index, 'mtBid'] = mt_bid
        replay_df.loc[step_target_index, 'bid'] = mt_bid

        step_target_data = replay_df.loc[step_target_index].copy()
        lambda2 = calculate_zt_bid_coefficient(step_target_data, zt_bgtleft=zt_bgtleft, m=m)
        replay_df.loc[step_target_index, 'ztBidCoefficient'] = lambda2
        zt_bid = pd.Series(0.0, index=step_target_index)
        if lambda2 > 0:
            candidate_mask = (
                (step_target_data['pValue'] > 0)
                & (step_target_data['bid'] > 0)
                & (step_target_data['leastWinningBid'] > step_target_data['bid'])
            )
            threshold = step_target_data['pValue'] / (
                m * (step_target_data['leastWinningBid'] - step_target_data['bid'])
            )
            selected_mask = candidate_mask & (threshold >= lambda2)
            zt_bid.loc[selected_mask[selected_mask].index] = step_target_data.loc[selected_mask, 'pValue'] / (
                m * lambda2
            )

        replay_df.loc[step_target_index, 'ztBid'] = zt_bid
        replay_df.loc[step_target_index, 'bid'] = mt_bid + zt_bid

        step_auction_mask = replay_df['timeStepIndex'].eq(time_step)
        _update_exposure_results(
            replay_df,
            step_auction_mask,
            auction_keys,
            step_target_mask,
            mt_lambda1,
            lambda2,
            m,
            zt_conversion_rate,
            random_generator
        )

        target_wins = replay_df.index[step_target_mask & replay_df['isExposed'].eq(1)]
        mt_spend = pd.to_numeric(replay_df.loc[target_wins, 'mt_cost'], errors='coerce').fillna(0.0).sum()
        zt_spend = pd.to_numeric(replay_df.loc[target_wins, 'zt_cost'], errors='coerce').fillna(0.0).sum()
        mt_bgtleft = max(mt_bgtleft - mt_spend, 0.0)
        zt_bgtleft = max(zt_bgtleft - zt_spend, 0.0)

        replay_df.loc[step_target_index, 'remainingBudget'] = mt_bgtleft
        replay_df.loc[step_target_index, 'ztRemainingBudget'] = zt_bgtleft
        replay_df.loc[step_target_index, 'isEnd'] = 1 if mt_bgtleft <= 0 or zt_bgtleft <= 0 else 0

    return replay_df


def _calculate_lambda1(time_step_data, mt_bgtleft):
    total_value = time_step_data['pValue'].sum()
    if total_value <= 0 or mt_bgtleft <= 0:
        return 0.0
    return total_value / mt_bgtleft


def _calculate_zt_cost_ratio(lambda1, lambda2, m):
    if lambda1 <= 0 or lambda2 <= 0:
        return 0.0

    c_y = m * lambda2
    denominator = (lambda1 - lambda2) * (lambda1 + c_y)
    if denominator <= 0:
        return 0.0

    rho = lambda1 ** 2 / denominator
    return min(max(rho, 0.0), 1.0)


def _add_least_winning_bid(df, auction_keys):
    auction_data = df[auction_keys].copy()
    auction_data['bid'] = pd.to_numeric(df['bid'], errors='coerce')
    ranked_bids = auction_data.sort_values(auction_keys + ['bid'], ascending=[True, True, True, False])
    ranked_bids['bid_rank_position'] = ranked_bids.groupby(auction_keys).cumcount() + 1
    threshold_bids = ranked_bids[ranked_bids['bid_rank_position'].isin([3, 4])]
    threshold_bids = threshold_bids.pivot_table(
        index=auction_keys,
        columns='bid_rank_position',
        values='bid',
        aggfunc='first'
    ).rename(columns={3: 'third_bid', 4: 'fourth_bid'}).reset_index()
    for col in ['third_bid', 'fourth_bid']:
        if col not in threshold_bids.columns:
            threshold_bids[col] = pd.NA

    auction_data = auction_data.merge(threshold_bids, on=auction_keys, how='left')
    df['leastWinningBid'] = auction_data['third_bid'].where(
        auction_data['bid'] < auction_data['third_bid'],
        auction_data['fourth_bid']
    ).fillna(0.0)
    return df


def _sample_conversion(index, conversion_rate, random_generator):
    return pd.Series(
        [1 if random_generator.random() < conversion_rate else 0 for _ in range(len(index))],
        index=index
    )


def _update_exposure_results(
        df,
        auction_mask,
        auction_keys,
        target_mask,
        lambda1,
        lambda2,
        m,
        zt_conversion_rate,
        random_generator):
    original_exposed = df.loc[auction_mask, 'originalIsExposed'].copy()
    original_conversion = df.loc[auction_mask, 'conversionAction'].copy()

    _add_least_winning_bid(df, auction_keys)
    auction_data = df.loc[auction_mask, auction_keys + ['bid']].copy()
    ranked = auction_data.sort_values(auction_keys + ['bid'], ascending=[True, True, True, False])
    ranked['bid_rank_position'] = ranked.groupby(auction_keys).cumcount() + 1
    exposed_index = ranked[ranked['bid_rank_position'] <= 3].index

    df.loc[auction_mask, 'isExposed'] = 0
    df.loc[exposed_index, 'isExposed'] = 1
    df.loc[auction_mask, ['mt_cost', 'zt_cost', 'total_cost']] = 0.0
    df.loc[auction_mask, 'conversionAction'] = 0

    total_cost = pd.to_numeric(df.loc[exposed_index, 'leastWinningBid'], errors='coerce').fillna(0.0)
    df.loc[exposed_index, 'total_cost'] = total_cost
    df.loc[exposed_index, 'mt_cost'] = total_cost

    zt_exposed_index = exposed_index.intersection(df.index[target_mask & df['ztBid'].gt(0)])
    if len(zt_exposed_index) > 0:
        zt_cost_ratio = _calculate_zt_cost_ratio(lambda1, lambda2, m)
        zt_cost = pd.to_numeric(df.loc[zt_exposed_index, 'total_cost'], errors='coerce').fillna(0.0) * zt_cost_ratio
        df.loc[zt_exposed_index, 'zt_cost'] = zt_cost
        df.loc[zt_exposed_index, 'mt_cost'] = df.loc[zt_exposed_index, 'total_cost'] - zt_cost

    kept_exposed_index = original_exposed[original_exposed.eq(1)].index.intersection(exposed_index)
    df.loc[kept_exposed_index, 'conversionAction'] = original_conversion.loc[kept_exposed_index]

    target_exposed_index = exposed_index.intersection(df.index[target_mask])
    if len(target_exposed_index) > 0:
        df.loc[target_exposed_index, 'conversionAction'] = _sample_conversion(
            target_exposed_index,
            zt_conversion_rate,
            random_generator
        )
