import missingno as msno
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

BETTING_HOUSES = {'p1_B365', 'p1_B&W', 'p1_CB', 'p1_EX', 'p1_LB', 'p1_GB', 'p1_IW', 'p1_PS', 'p1_SB', 'p1_SJ', 'p1_UB',
                  'p1_Max', 'p1_Avg', 'p2_B365', 'p2_B&W', 'p2_CB', 'p2_EX', 'p2_LB', 'p2_GB', 'p2_IW', 'p2_PS',
                  'p2_SB', 'p2_SJ', 'p2_UB', 'p2_Max', 'p2_Avg'}


def print_missing_data():
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        print(data.columns.values)
        msno.matrix(data)
        data.apply(lambda x: 43 - x.count(), axis=1).value_counts()


def find_correlation():
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        train, test = train_test_split(data, train_size=0.8, random_state=11)
        corr_matrix = train.corr()
        #fig, ax = plt.subplots(figsize=(18, 13))
        #sns.heatmap(corr_matrix, annot_kws={"size": 20})
        #plt.title(year)
        #plt.show()

        kot = corr_matrix[corr_matrix >= .9]
        plt.figure(figsize=(12, 10))
        sns.heatmap(kot, cmap="Greens")
        plt.title(f'{year} : corr_matrix >= .9')
        plt.show()

        print(f'{year}:')
        corr_mtrx = get_top_abs_correlations(train)

        #print_corr_between_two_features(train, 'p1_svpt', 'p2_2ndWon', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_elo_bestRank', 'p1_elo_rank', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_elo_bestRank', 'p2_elo_rank', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_elo_points', 'p1_elo_rank', corr_mtrx)

        #print_corr_between_two_features(train, 'p1_svpt', 'p1_1stWon', corr_mtrx)

        #print_corr_between_two_features(train, 'p1_svpt', 'p1_1stIn', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_svpt', 'p1_1stWon', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_svpt', 'p2_svpt', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_svpt', 'p1_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_svpt', 'p2_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_1stWon', 'p1_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_1stIn', 'p1_1stWon', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_1stIn', 'p1_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_1stIn', 'p2_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_SvGms', 'p2_svpt', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_SvGms', 'p2_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_svpt', 'p2_1stIn', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_svpt', 'p2_SvGms', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_1stIn', 'p2_1stWon', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_bpSaved', 'p2_bpFaced', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_GB', 'p2_IW', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_CB', 'p2_GB', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_GB', 'p1_IW', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_GB', 'p2_IW', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_CB', 'p2_IW', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_B365', 'p2_CB', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_B365', 'p2_EX', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_B365', 'p2_PS', corr_mtrx)
        ##print_corr_between_two_features(train, 'p2_CB', 'p2_EX', corr_mtrx)
        #print_corr_between_two_features(train, 'p2_CB', 'p2_PS', corr_mtrx)
        #print_corr_between_two_features(train, 'p1_B365', 'p1_CB')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_IW')
        #print_corr_between_two_features(train, 'p2_CB', 'p2_IW')
        #print_corr_between_two_features(train, 'p2_PS', 'p2_UB')
        #print_corr_between_two_features(train, 'p1_CB', 'p1_EX')
        #print_corr_between_two_features(train, 'p1_PS', 'p1_UB')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_LB')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_UB')
        #print_corr_between_two_features(train, 'p2_LB', 'p2_EX')
        #print_corr_between_two_features(train, 'p2_UB', 'p2_EX')
        #print_corr_between_two_features(train, 'p2_UB', 'p2_LB')
        #print_corr_between_two_features(train, 'p1_UB', 'p1_LB')
        #print_corr_between_two_features(train, 'p2_UB', 'p2_PS')
        #print_corr_between_two_features(train, 'p1_UB', 'p1_PS')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_EX')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_LB')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_SJ')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_UB')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_SJ')
        #print_corr_between_two_features(train, 'p1_LB', 'p1_EX')
        #print_corr_between_two_features(train, 'p1_SJ', 'p1_EX')
        #print_corr_between_two_features(train, 'p1_UB', 'p1_EX')
        #print_corr_between_two_features(train, 'p1_SJ', 'p2_EX')
        #print_corr_between_two_features(train, 'p1_LB', 'p1_SJ')
        #print_corr_between_two_features(train, 'p2_LB', 'p2_SJ')
        #print_corr_between_two_features(train, 'p1_UB', 'p1_SJ')
        #print_corr_between_two_features(train, 'p2_UB', 'p2_SJ')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_PS')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_Avg')
        #print_corr_between_two_features(train, 'p1_LB', 'p1_PS')
        #print_corr_between_two_features(train, 'p2_LB', 'p2_Avg')
        #print_corr_between_two_features(train, 'p1_SJ', 'p1_PS')
        #print_corr_between_two_features(train, 'p2_Avg', 'p2_SJ')
        #print_corr_between_two_features(train, 'p1_Max', 'p1_Avg')
        #print_corr_between_two_features(train, 'p2_Max', 'p2_Avg')
        #print_corr_between_two_features(train, 'p1_set5_score', 'p2_set5_score')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_Avg')
        #print_corr_between_two_features(train, 'p2_B365', 'p2_Max')
        #print_corr_between_two_features(train, 'p1_Avg', 'p1_EX')
        #print_corr_between_two_features(train, 'p2_Avg', 'p2_EX')
        #print_corr_between_two_features(train, 'p1_LB', 'p1_Avg')
        #print_corr_between_two_features(train, 'p1_Max', 'p1_PS')
        #print_corr_between_two_features(train, 'p1_Avg', 'p1_PS')
        #print_corr_between_two_features(train, 'p2_Max', 'p2_PS')
        #print_corr_between_two_features(train, 'p2_Avg', 'p2_PS')
        #print_corr_between_two_features(train, 'p1_B365', 'p1_Max')
        #print_corr_between_two_features(train, 'p1_LB', 'p1_Max')
        #print_corr_between_two_features(train, 'p2_SJ', 'p2_PS')
        #print_corr_between_two_features(train, 'p1_SJ', 'p1_Avg')
        #print_corr_between_two_features(train, 'p2_LB', 'p2_PS')
        #print_corr_between_two_features(train, 'p2_LB', 'p2_Max')
        #print_corr_between_two_features(train, 'p2_PS', 'p2_EX')


def get_top_abs_correlations(df):
    au_corr = df.corr().abs().unstack().drop_duplicates()
    print(au_corr[au_corr > 0.92])
    print("***************")
    print(au_corr['p1_1stWon'])
    print('*****************')
    return au_corr


def print_all_features():
    set_of_cols = set()
    for year in range(2001, 2022):
        tbl = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        set_of_cols.update(set(tbl.columns.values))
    print(set_of_cols)


def print_corr_between_two_features(data, f1, f2, corr):
    labels = data.columns.values
    if f1 in labels and f2 in labels:
        g = sns.jointplot(data=data, x=f1, y=f2, hue="p1_won")
        g.ax_joint.grid()
        g.fig.suptitle(f'corr = {corr[f1, f2]}')
        plt.show()


def plot_specific_feature(data: pd.DataFrame, feature: str, year: int, subtitle=None):
    sns.histplot(data[feature], bins=56, kde=True).set(title=f'{feature} - {year}')
    plt.suptitle(subtitle)
    plt.grid()
    plt.show()


def normalaize_betting_odds_features():
    betting_features_by_year_by_player = dict()
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        train, test = train_test_split(data, train_size=0.8, random_state=11)
        all_features = train.columns.values
        betting_features_by_year_by_player[f'{year}_p1'] = []
        betting_features_by_year_by_player[f'{year}_p2'] = []
        mean_vals_dict = dict()
        std_vals_dict = dict()
        for feature in all_features:
            if feature in BETTING_HOUSES:
                if 'p1' in feature:
                    betting_features_by_year_by_player[f'{year}_p1'].append(feature)
                else:
                    betting_features_by_year_by_player[f'{year}_p2'].append(feature)
                print(year, feature, train[feature].isna().sum())
                #plot_specific_feature(train, feature, year,  "Before Normalize")

                mean = mean_vals_dict[feature] = train[feature].mean(skipna=True)
                std = std_vals_dict[feature] = train[feature].std(skipna=True)
                train[feature] = (train[feature] - mean) / std

                #plot_specific_feature(train, feature, year, "After calc z-score norm")

        print(betting_features_by_year_by_player[f'{year}_p1'])
        print(betting_features_by_year_by_player[f'{year}_p2'])

        print(f'{year}, {mean_vals_dict=}')
        print(f'{year}, {std_vals_dict=}')

        train['p1_betting_odds'] = train[betting_features_by_year_by_player[f'{year}_p1']].mean(axis=1)
        train['p2_betting_odds'] = train[betting_features_by_year_by_player[f'{year}_p2']].mean(axis=1)

        plot_specific_feature(train, 'p1_betting_odds', year, "Final betting odds feature")
        plot_specific_feature(train, 'p2_betting_odds', year, "Final betting odds feature")

        train_with_avg_betting_odds_only = train.drop(betting_features_by_year_by_player[f'{year}_p1'], axis='columns').copy()
        train_with_avg_betting_odds_only = train_with_avg_betting_odds_only.drop(betting_features_by_year_by_player[f'{year}_p2'], axis='columns').copy()

        train_with_avg_betting_odds_only.to_csv(f'Data/train/only_avg_betting_odds_{year}.csv', index=False)


def drop_unsued_features():
    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/only_avg_betting_odds_{year}.csv')
        print(f'{year} Before drop, {len(train.columns.values)}')

        candidates = ['p1_Pts', 'p2_Pts', 'p1_elo_points', 'p2_elo_points', 'p2_elo_bestPoints', 'tourney_name',
                      'p1_elo_bestPoints', 'p1_1stIn', 'p2_1stIn', 'tourney_id', 'match_num', 'Court']
        train = train.drop([x for x in candidates if x in train.columns], axis='columns').copy()

        train.to_csv(f'Data/train/more_dropped_data_{year}.csv', index=False)

        print(f'{year} After drop, {len(train.columns.values)}')


def transform_data_values():
    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/more_dropped_data_{year}.csv')

        # transform surface to one-hot-encoding
        train['carpet'] = np.where((train['surface'] == 'Carpet'), 1, -1)
        train['clay'] = np.where((train['surface'] == 'Clay'), 1, -1)
        train['grass'] = np.where((train['surface'] == 'Grass'), 1, -1)
        train['hard'] = np.where((train['surface'] == 'Hard'), 1, -1)
        train = train.drop('surface', axis='columns').copy()

        # Normalaize draw_size by min-max
        min_vals_dict = dict()
        max_vals_dict = dict()
        features_to_have_min_max = ['draw_size', 'minutes', 'p1_ace', 'p1_df', 'p1_svpt', 'p1_1stWon', 'p1_2ndWon',
                                    'p1_SvGms', 'p1_bpSaved', 'p1_bpFaced',	'p2_ace', 'p2_df', 'p2_svpt', 'p2_1stWon',
                                    'p2_2ndWon', 'p2_SvGms', 'p2_bpSaved', 'p2_bpFaced', 'p1_atp_rank',
                                    'p1_atp_rank_points', 'p2_atp_rank', 'p2_atp_rank_points', 'p1_elo_rank',
                                    'p1_elo_bestRank', 'p2_elo_rank', 'p2_elo_bestRank', 'p1_set1_score',
                                    'p2_set1_score', 'set1_breakpoint_score', 'p1_set2_score', 'p2_set2_score',
                                    'set2_breakpoint_score', 'p1_set3_score', 'p2_set3_score', 'set3_breakpoint_score',
                                    'p1_set4_score', 'p2_set4_score', 'set4_breakpoint_score', 'p1_set5_score',
                                    'p2_set5_score', 'set5_breakpoint_score']

        for feature in features_to_have_min_max:
            min_val = min_vals_dict[feature] = train[feature].min()
            max_val = max_vals_dict[feature] = train[feature].max()
            train[feature] = (train[feature] - min_val) / (max_val - min_val)
        print(train['draw_size'])

        print(f'{year}, {min_vals_dict=}')
        print(f'{year}, {max_vals_dict=}')

        #transform tourney_level to one-hot-encoding
        train['masters_1000s'] = np.where((train['tourney_level'] == 'M'), 1, -1)
        train['grand_slams'] = np.where((train['tourney_level'] == 'G'), 1, -1)
        train['other_tour-level'] = np.where((train['tourney_level'] == 'A'), 1, -1)
        train['challengers'] = np.where((train['tourney_level'] == 'C'), 1, -1)
        train['satellites_ITFs'] = np.where((train['tourney_level'] == 'S'), 1, -1)
        train['tour_finals'] = np.where((train['tourney_level'] == 'F'), 1, -1)
        train['davis_cup'] = np.where((train['tourney_level'] == 'D'), 1, -1)
        train = train.drop('tourney_level', axis='columns').copy()

        #transform pi_hand to binary number (R=1, L=-1)
        train['p1_hand'] = np.where((train['p1_hand'] == 'R'), 1, -1)
        train['p2_hand'] = np.where((train['p2_hand'] == 'R'), 1, -1)

        # run z-score normalize on numeric features
        mean_vals_dict = dict()
        std_vals_dict = dict()
        features_to_have_z_score = ['p1_ht', 'p2_ht', 'p1_age', 'p2_age']

        for feature in features_to_have_z_score:
            mean = mean_vals_dict[feature] = train[feature].mean()
            std = std_vals_dict[feature] = train[feature].std()
            train[feature] = (train[feature] - mean) / std

        print(f'{year}, {mean_vals_dict=}')
        print(f'{year}, {std_vals_dict=}')

        # transform best_of to binary number (3=1, 5=-1)
        train['best_of'] = np.where((train['best_of'] == 3), 1, -1)

        # transform round to one-hot-encoding
        train['f_round'] = np.where((train['round'] == 'F'), 1, -1)
        train['qf_round'] = np.where((train['round'] == 'QF'), 1, -1)
        train['r128_round'] = np.where((train['round'] == 'R128'), 1, -1)
        train['r16_round'] = np.where((train['round'] == 'R16'), 1, -1)
        train['r32_round'] = np.where((train['round'] == 'R32'), 1, -1)
        train['r64_round'] = np.where((train['round'] == 'R64'), 1, -1)
        train['rr_round'] = np.where((train['round'] == 'RR'), 1, -1)
        train['sf_round'] = np.where((train['round'] == 'SF'), 1, -1)
        train = train.drop('round', axis='columns').copy()

        # transform p1_won to signed 1/-1 instead of unsigned 1/0
        train['p1_won'] = np.where((train['p1_won'] == 1), 1, -1)

        train.to_csv(f'Data/train/after_first_transformation_{year}.csv', index=False)


def remove_extra_chars_from_sets():
    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/more_dropped_data_{year}.csv')
        sets_scores_features = ['p1_set1_score', 'p2_set1_score', 'set1_breakpoint_score', 'p1_set2_score',
                                'p2_set2_score', 'set2_breakpoint_score', 'p1_set3_score', 'p2_set3_score',
                                'set3_breakpoint_score', 'p1_set4_score', 'p2_set4_score', 'set4_breakpoint_score',
                                'p1_set5_score', 'p2_set5_score', 'set5_breakpoint_score']

        for feature in sets_scores_features:
            print(feature)
            for index, row in train.iterrows():
                if isinstance(row[feature], str) and '[' in row[feature]:
                    new_val = row[feature][1:]
                    train.at[index, feature] = new_val
                if isinstance(row[feature], str) and ']' in row[feature]:
                    new_val = row[feature][:-1]
                    train.at[index, feature] = new_val

        train.to_csv(f'Data/train/more_dropped_data_{year}.csv', index=False)


def fill_missing_data():
    features_with_missing_vals = {'p2_elo_bestRank', 'minutes', 'p1_atp_rank', 'p1_1stWon', 'p1_2ndWon', 'p2_svpt',
                                  'p2_bpFaced', 'p2_df', 'p2_ace', 'p2_1stWon', 'p1_ace', 'p2_atp_rank_points',
                                  'p1_svpt', 'p1_SvGms', 'p1_elo_bestRank', 'p1_atp_rank_points', 'p1_elo_rank',
                                  'p2_elo_rank', 'p1_bpSaved', 'p2_SvGms', 'p1_betting_odds', 'p2_2ndWon',
                                  'p2_bpSaved', 'p2_ht', 'p1_ht', 'p1_bpFaced', 'p1_df', 'p2_atp_rank',
                                  'p2_betting_odds', 'p1_age', 'p2_age'}

    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/after_first_transformation_{year}.csv')
        features_mean = dict()
        for feature in features_with_missing_vals:
            features_mean[feature] = train[feature].mean()
        train = train.fillna(features_mean)
        print(year, features_mean)

        train.to_csv(f'Data/train/after_fill_missing_vals_{year}.csv', index=False)


def transform_dates():
    first_date = pd.to_datetime('01/01/2001')
    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/after_fill_missing_vals_{year}.csv')
        train['tourney_date'] = train['tourney_date'].apply(lambda x: pd.to_datetime(x))
        train['tourney_date'] = train['tourney_date'].apply(lambda x: (x - first_date).days)
        print(train['tourney_date'])
        train.to_csv(f'Data/train/with_numeric_date_{year}.csv', index=False)


def main():
    #print_all_features()
    #find_correlation()
    #normalaize_betting_odds_features()
    #drop_unsued_features()
    #remove_extra_chars_from_sets()
    #transform_data_values()
    #fill_missing_data()
    transform_dates()


if __name__ == '__main__':
    main()