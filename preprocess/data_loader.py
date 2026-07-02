import os
import pandas as pd
import warnings
import glob

from auction_replay import replay_single_advertiser_zt
from zt_bid_coefficient import calculate_zt_bid_coefficient

warnings.filterwarnings('ignore')


class RlDataGenerator:
    def __init__(self, file_folder_path="./data/traffic"):
        self.file_folder_path = file_folder_path
        self.training_data_path = self.file_folder_path + "/" + "training_data_rlData_folder"
        self.replay_data_path = self.file_folder_path + "/" + "zt_replay_data_folder"

    @staticmethod
    def _calculate_mt_bid_coefficient(time_step_data):
        total_bid = time_step_data['bid'].sum()
        total_value = time_step_data['pValue'].sum()
        return total_bid / total_value if total_value > 0 else 0

    @staticmethod
    def _add_least_winning_bid(df):
        auction_keys = ['deliveryPeriodIndex', 'timeStepIndex', 'pvIndex']

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

    def batch_generate_rl_data(self):
        os.makedirs(self.training_data_path, exist_ok=True)
        os.makedirs(self.replay_data_path, exist_ok=True)
        csv_files = glob.glob(os.path.join(self.file_folder_path, '*.csv'))
        print(csv_files)
        training_data_list = []
        for csv_path in csv_files:
            print("开始处理文件：", csv_path)
            df = pd.read_csv(csv_path)
            csv_filename = os.path.basename(csv_path)
            for advertiserNumber in sorted(df['advertiserNumber'].unique()):
                replay_df = replay_single_advertiser_zt(df, advertiserNumber)
                advertiser_label = str(advertiserNumber).replace('.', '_')
                replay_filename = csv_filename.replace('.csv', f'-ztReplay-advertiser-{advertiser_label}.csv')
                replay_path = os.path.join(self.replay_data_path, replay_filename)
                replay_df.to_csv(replay_path, index=False)

                df_processed = self._generate_rl_data(replay_df)
                trainData_filename = csv_filename.replace('.csv', f'-advertiser-{advertiser_label}-rlData.csv')
                trainData_path = os.path.join(self.training_data_path, trainData_filename)
                df_processed.to_csv(trainData_path, index=False)
                training_data_list.append(df_processed)
                del replay_df, df_processed
            print("处理文件成功：", csv_path)
            del df
        combined_dataframe = pd.concat(training_data_list, axis=0, ignore_index=True)
        combined_dataframe_path = os.path.join(self.training_data_path, "training_data_all-rlData.csv")
        combined_dataframe.to_csv(combined_dataframe_path, index=False)
        print("整合多天训练数据成功；保存至:", combined_dataframe_path)

    def _generate_rl_data(self, df):
        df = self._add_least_winning_bid(df)
        training_data_rows = []
        for (
                deliveryPeriodIndex, advertiserNumber, advertiserCategoryIndex, mt_budget,
                CPAConstraint), group in df.groupby(
            ['deliveryPeriodIndex', 'advertiserNumber', 'advertiserCategoryIndex', 'budget', 'CPAConstraint']):
            group = group.sort_values('timeStepIndex')
            group['timeStepIndex_volume'] = group.groupby('timeStepIndex')['timeStepIndex'].transform('size')
            timeStepIndex_volume_sum = group.groupby('timeStepIndex')['timeStepIndex_volume'].first()
            historical_volume = timeStepIndex_volume_sum.cumsum().shift(1).fillna(0).astype(int)
            group['historical_volume'] = group['timeStepIndex'].map(historical_volume)
            last_3_timeStepIndexs_volume = timeStepIndex_volume_sum.rolling(window=3, min_periods=1).sum().shift(
                1).fillna(0).astype(int)
            group['last_3_timeStepIndexs_volume'] = group['timeStepIndex'].map(last_3_timeStepIndexs_volume)
            group_agg = group.groupby('timeStepIndex').agg({
                'bid': 'mean',
                'leastWinningCost': 'mean',
                'conversionAction': 'mean',
                'xi': 'mean',
                'pValue': 'mean',
                'timeStepIndex_volume': 'first'
            }).reset_index()
            for col in ['bid', 'leastWinningCost', 'conversionAction', 'xi', 'pValue']:
                group_agg[f'avg_{col}_all'] = group_agg[col].expanding().mean().shift(1)
                group_agg[f'avg_{col}_last_3'] = group_agg[col].rolling(window=3, min_periods=1).mean().shift(1)
            group = group.merge(group_agg, on='timeStepIndex', suffixes=('', '_agg'))
            realAllCost = (group['isExposed'] * group['total_cost']).sum()
            realAllConversion = group['conversionAction'].sum()
            for timeStepIndex in group['timeStepIndex'].unique():
                current_timeStepIndex_data = group[group['timeStepIndex'] == timeStepIndex]
                timeStepIndexNum = 48
                current_timeStepIndex_data.fillna(0, inplace=True)
                mt_budget = current_timeStepIndex_data['budget'].iloc[0]
                zt_budget = current_timeStepIndex_data['ztBudget'].iloc[
                    0] if 'ztBudget' in current_timeStepIndex_data.columns else mt_budget
                remainingBudget = current_timeStepIndex_data['remainingBudget'].iloc[0]
                timeleft = (timeStepIndexNum - timeStepIndex) / timeStepIndexNum
                mt_bgtleft = remainingBudget
                zt_bgtleft = current_timeStepIndex_data['ztRemainingBudget'].iloc[
                    0] if 'ztRemainingBudget' in current_timeStepIndex_data.columns else remainingBudget
                state_features = current_timeStepIndex_data.iloc[0].to_dict()
                mt_bid_coefficient = self._calculate_mt_bid_coefficient(current_timeStepIndex_data)
                state = (
                    timeleft, mt_bgtleft, zt_bgtleft,
                    mt_bid_coefficient,
                    state_features['avg_bid_all'],
                    state_features['avg_bid_last_3'],
                    state_features['avg_leastWinningCost_all'],
                    state_features['avg_pValue_all'],
                    state_features['avg_conversionAction_all'],
                    state_features['avg_xi_all'],
                    state_features['avg_leastWinningCost_last_3'],
                    state_features['avg_pValue_last_3'],
                    state_features['avg_conversionAction_last_3'],
                    state_features['avg_xi_last_3'],
                    state_features['pValue_agg'],
                    state_features['timeStepIndex_volume_agg'],
                    state_features['last_3_timeStepIndexs_volume'],
                    state_features['historical_volume']
                )

                if 'ztBidCoefficient' in current_timeStepIndex_data.columns:
                    action = current_timeStepIndex_data['ztBidCoefficient'].iloc[0]
                else:
                    action = calculate_zt_bid_coefficient(
                        current_timeStepIndex_data,
                        zt_bgtleft=zt_bgtleft
                    )
                reward = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'conversionAction'].sum()
                reward_continuous = current_timeStepIndex_data[current_timeStepIndex_data['isExposed'] == 1][
                    'pValue'].sum()

                done = 1 if timeStepIndex == timeStepIndexNum - 1 or current_timeStepIndex_data['isEnd'].iloc[
                    0] == 1 else 0

                training_data_rows.append({
                    'ztAdvertiserNumber': current_timeStepIndex_data['ztAdvertiserNumber'].iloc[
                        0] if 'ztAdvertiserNumber' in current_timeStepIndex_data.columns else None,
                    'deliveryPeriodIndex': deliveryPeriodIndex,
                    'advertiserNumber': advertiserNumber,
                    'advertiserCategoryIndex': advertiserCategoryIndex,
                    'mt_budget': mt_budget,
                    'zt_budget': zt_budget,
                    'CPAConstraint': CPAConstraint,
                    'realAllCost':realAllCost,
                    'realAllConversion': realAllConversion,
                    'timeStepIndex': timeStepIndex,
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'reward_continuous': reward_continuous,
                    'done': done
                })
        training_data = pd.DataFrame(training_data_rows)
        training_data = training_data.sort_values(by=['deliveryPeriodIndex', 'advertiserNumber', 'timeStepIndex'])

        training_data['next_state'] = training_data.groupby(['deliveryPeriodIndex', 'advertiserNumber'])['state'].shift(
            -1)
        training_data.loc[training_data['done'] == 1, 'next_state'] = None
        return training_data


def generate_rl_data():
    file_folder_path = "../data/data_AuctionNet/traffic"
    data_loader = RlDataGenerator(file_folder_path=file_folder_path)
    data_loader.batch_generate_rl_data()


if __name__ == '__main__':
    generate_rl_data()
