import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from scipy import stats

BETTING_HOUSES = {'p1_B365', 'p1_B&W', 'p1_CB', 'p1_EX', 'p1_LB', 'p1_GB', 'p1_IW', 'p1_PS', 'p1_SB', 'p1_SJ', 'p1_UB',
                  'p1_Max', 'p1_Avg', 'p2_B365', 'p2_B&W', 'p2_CB', 'p2_EX', 'p2_LB', 'p2_GB', 'p2_IW', 'p2_PS',
                  'p2_SB', 'p2_SJ', 'p2_UB', 'p2_Max', 'p2_Avg'}


def print_corr():
    """
    For every year plot heatmap of correlations between features.
    """
    for year in range(2001, 2022):
        dataset = pd.read_csv('Data/p1p2_after_mismatch{year}.csv')

        train, test = train_test_split(dataset, train_size=0.8, random_state=11)
        corr_matrix = train.corr()
        fig, ax = plt.subplots(figsize=(18, 13))
        sns.heatmap(corr_matrix, annot_kws={"size": 20})
        plt.title(year)
        plt.show()

        print(corr_matrix)
        plt.show()


def find_correlation():
    """
    For every year plot heatmap of correlations between features. with option to plot scatter plot between two features
    that are high correlated.
    """
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        train, test = train_test_split(data, train_size=0.8, random_state=11)
        corr_matrix = train.corr()
        fig, ax = plt.subplots(figsize=(18, 13))
        sns.heatmap(corr_matrix, annot_kws={"size": 20})
        plt.title(year)
        plt.show()

        kot = corr_matrix[corr_matrix >= .9]
        plt.figure(figsize=(12, 10))
        sns.heatmap(kot, cmap="Greens")
        plt.title(f'{year} : corr_matrix >= .9')
        plt.show()

        print(f'{year}:')
        corr_mtrx = get_top_abs_correlations(train)

        # print_corr_between_two_features(train, 'p1_svpt', 'p2_2ndWon', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_elo_bestRank', 'p1_elo_rank', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_elo_bestRank', 'p2_elo_rank', corr_mtrx)
        print_corr_between_two_features(train, 'p1_elo_points', 'p1_elo_rank', corr_mtrx)

        # print_corr_between_two_features(train, 'p1_svpt', 'p1_1stWon', corr_mtrx)

        # print_corr_between_two_features(train, 'p1_svpt', 'p1_1stIn', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_svpt', 'p1_1stWon', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_svpt', 'p2_svpt', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_svpt', 'p1_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_svpt', 'p2_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_1stWon', 'p1_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_1stIn', 'p1_1stWon', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_1stIn', 'p1_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_1stIn', 'p2_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_SvGms', 'p2_svpt', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_SvGms', 'p2_SvGms', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_svpt', 'p2_1stIn', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_svpt', 'p2_SvGms', corr_mtrx)
        #  print_corr_between_two_features(train, 'p2_1stIn', 'p2_1stWon', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_bpSaved', 'p2_bpFaced', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_GB', 'p2_IW', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_CB', 'p2_GB', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_GB', 'p1_IW', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_GB', 'p2_IW', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_CB', 'p2_IW', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_B365', 'p2_CB', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_B365', 'p2_EX', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_B365', 'p2_PS', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_CB', 'p2_EX', corr_mtrx)
        # print_corr_between_two_features(train, 'p2_CB', 'p2_PS', corr_mtrx)
        # print_corr_between_two_features(train, 'p1_B365', 'p1_CB')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_IW')
        # print_corr_between_two_features(train, 'p2_CB', 'p2_IW')
        # print_corr_between_two_features(train, 'p2_PS', 'p2_UB')
        # print_corr_between_two_features(train, 'p1_CB', 'p1_EX')
        # print_corr_between_two_features(train, 'p1_PS', 'p1_UB')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_LB')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_UB')
        # print_corr_between_two_features(train, 'p2_LB', 'p2_EX')
        # print_corr_between_two_features(train, 'p2_UB', 'p2_EX')
        # print_corr_between_two_features(train, 'p2_UB', 'p2_LB')
        # print_corr_between_two_features(train, 'p1_UB', 'p1_LB')
        # print_corr_between_two_features(train, 'p2_UB', 'p2_PS')
        # print_corr_between_two_features(train, 'p1_UB', 'p1_PS')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_EX')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_LB')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_SJ')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_UB')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_SJ')
        # print_corr_between_two_features(train, 'p1_LB', 'p1_EX')
        # print_corr_between_two_features(train, 'p1_SJ', 'p1_EX')
        # print_corr_between_two_features(train, 'p1_UB', 'p1_EX')
        # print_corr_between_two_features(train, 'p1_SJ', 'p2_EX')
        # print_corr_between_two_features(train, 'p1_LB', 'p1_SJ')
        # print_corr_between_two_features(train, 'p2_LB', 'p2_SJ')
        # print_corr_between_two_features(train, 'p1_UB', 'p1_SJ')
        # print_corr_between_two_features(train, 'p2_UB', 'p2_SJ')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_PS')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_Avg')
        # print_corr_between_two_features(train, 'p1_LB', 'p1_PS')
        # print_corr_between_two_features(train, 'p2_LB', 'p2_Avg')
        # print_corr_between_two_features(train, 'p1_SJ', 'p1_PS')
        # print_corr_between_two_features(train, 'p2_Avg', 'p2_SJ')
        # print_corr_between_two_features(train, 'p1_Max', 'p1_Avg')
        # print_corr_between_two_features(train, 'p2_Max', 'p2_Avg')
        # print_corr_between_two_features(train, 'p1_set5_score', 'p2_set5_score')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_Avg')
        # print_corr_between_two_features(train, 'p2_B365', 'p2_Max')
        # print_corr_between_two_features(train, 'p1_Avg', 'p1_EX')
        # print_corr_between_two_features(train, 'p2_Avg', 'p2_EX')
        # print_corr_between_two_features(train, 'p1_LB', 'p1_Avg')
        # print_corr_between_two_features(train, 'p1_Max', 'p1_PS')
        # print_corr_between_two_features(train, 'p1_Avg', 'p1_PS')
        # print_corr_between_two_features(train, 'p2_Max', 'p2_PS')
        # print_corr_between_two_features(train, 'p2_Avg', 'p2_PS')
        # print_corr_between_two_features(train, 'p1_B365', 'p1_Max')
        # print_corr_between_two_features(train, 'p1_LB', 'p1_Max')
        # print_corr_between_two_features(train, 'p2_SJ', 'p2_PS')
        # print_corr_between_two_features(train, 'p1_SJ', 'p1_Avg')
        # print_corr_between_two_features(train, 'p2_LB', 'p2_PS')
        # print_corr_between_two_features(train, 'p2_LB', 'p2_Max')
        # print_corr_between_two_features(train, 'p2_PS', 'p2_EX')


def get_top_abs_correlations(df):
    """
    print all features the the correlation between them is higher than 0.92.
    :param df: the full correlation matrix
    :return: correlation matrix filterd by correlations higher then 0.92
    """
    au_corr = df.corr().abs().unstack().drop_duplicates()
    print(au_corr[au_corr > 0.92])
    return au_corr


def print_all_features():
    """
    print all features through the years. (betting odds features changed from year to year.
    """
    set_of_cols = set()
    for year in range(2001, 2022):
        tbl = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        set_of_cols.update(set(tbl.columns.values))
    print(set_of_cols)


def print_corr_between_two_features(data: pd.DataFrame, f1: str, f2: str, corr: pd.DataFrame):
    """
    plot scatterplot of the relation between two features with specify of the target label.
    :param data: data of a year
    :param f1: feature 1 name
    :param f2: feature 2 name
    :param corr: the correlation of the two features
    """
    labels = data.columns.values
    if f1 in labels and f2 in labels:
        g = sns.jointplot(data=data, x=f1, y=f2, hue="p1_won")
        g.ax_joint.grid()
        g.fig.suptitle(f'corr = {corr[f1, f2]}')
        plt.show()


def plot_specific_feature(data: pd.DataFrame, feature: str, year: int, subtitle=None):
    """
    plot hostogram of a specific feature by year.
    :param data: all data of the year
    :param feature: feature name
    :param year: the year the data belong to
    :param subtitle: for the plot
    """
    sns.histplot(data[feature], bins=56, kde=True).set(title=f'{feature} - {year}')
    plt.suptitle(subtitle)
    plt.grid()
    plt.show()


def normalize_betting_odds_features():
    """
    Normalizes all betting odds feature. each feature of betting odds transform in z-score normalization, and then
    calculate the average of all features in the year to one feature, and drop the initial features.
    In first, split the data to train and test. All calculations are on train set only!
    :return: save the new table to csv file.
    """
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
                # plot_specific_feature(train, feature, year,  "Before Normalize")

                mean = mean_vals_dict[feature] = train[feature].mean(skipna=True)
                std = std_vals_dict[feature] = train[feature].std(skipna=True)
                train[feature] = (train[feature] - mean) / std

                # plot_specific_feature(train, feature, year, "After calc z-score norm")

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
    """
    Drop more features. details about the features in the report.
    :return: save the new table to csv file.
    """
    for year in range(2001, 2022):
        train = pd.read_csv(f'Data/train/only_avg_betting_odds_{year}.csv')
        print(f'{year} Before drop, {len(train.columns.values)}')

        candidates = ['p1_Pts', 'p2_Pts', 'p1_elo_points', 'p2_elo_points', 'p2_elo_bestPoints', 'tourney_name',
                      'p1_elo_bestPoints', 'p1_1stIn', 'p2_1stIn', 'tourney_id', 'match_num', 'Court']
        train = train.drop([x for x in candidates if x in train.columns], axis='columns').copy()

        train.to_csv(f'Data/train/more_dropped_data_{year}.csv', index=False)

        print(f'{year} After drop, {len(train.columns.values)}')


def remove_extra_chars_from_sets(folder='train'):
    """
    Some tomes there are extra chats in the scores features (that we split from one long string). This func find this
    chars and remove then.
    :param folder: where to save the new table: train or test.
    :return: save the new table to csv file.
    """
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/{folder}/more_dropped_data_{year}.csv')
        sets_scores_features = ['p1_set1_score', 'p2_set1_score', 'set1_breakpoint_score', 'p1_set2_score',
                                'p2_set2_score', 'set2_breakpoint_score', 'p1_set3_score', 'p2_set3_score',
                                'set3_breakpoint_score', 'p1_set4_score', 'p2_set4_score', 'set4_breakpoint_score',
                                'p1_set5_score', 'p2_set5_score', 'set5_breakpoint_score']

        for feature in sets_scores_features:
            print(feature)
            for index, row in data.iterrows():
                if isinstance(row[feature], str) and '[' in row[feature]:
                    new_val = row[feature][1:]
                    data.at[index, feature] = new_val
                if isinstance(row[feature], str) and ']' in row[feature]:
                    new_val = row[feature][:-1]
                    data.at[index, feature] = new_val

        data.to_csv(f'Data/{folder}/more_dropped_data_{year}.csv', index=False)


def create_all_years_uniform_data():
    """
    Join all first pre-processed data years together to one big table.
    :return: save the new table to csv file.
    """
    all_years = []
    for year in range(2001, 2022):
        all_years.append(pd.read_csv(f'Data/train/more_dropped_data_{year}.csv'))

    data_all_years = pd.concat(all_years)
    print(data_all_years.shape)
    print(data_all_years.head)
    data_all_years.to_csv('Data/train/primary_all_years.csv', index=False)


def prepare_all_years_data(only_z_score=False):
    """
    Transform the big table values. Categorical features with one-hot-encoding. Implement z-score normalization on
    features with standard distribution of the values, and implement min-max normalization to the others.
    print the mean, std, min and max value of each feature that was normalized, for saving the values for the test
    preparation data.
    Transform dates of the feature 'tourney_date' to numerical value: (amount of days after 01/01/2001) / 365
    Fill missing data of numerical features by put the mean value in the missing value places.
    :return: save the new table to csv file.
    """
    train = pd.read_csv('Data/train/primary_all_years.csv')

    # transform surface to one-hot-encoding
    train['carpet'] = np.where((train['surface'] == 'Carpet'), 1, -1)
    train['clay'] = np.where((train['surface'] == 'Clay'), 1, -1)
    train['grass'] = np.where((train['surface'] == 'Grass'), 1, -1)
    train['hard'] = np.where((train['surface'] == 'Hard'), 1, -1)
    train = train.drop('surface', axis='columns').copy()

    # Normalaize draw_size by min-max
    features_to_have_min_max = ['draw_size', 'p1_df', 'p1_bpSaved', 'p1_bpFaced', 'p2_df', 'p2_bpSaved', 'p2_bpFaced',
                                'p1_atp_rank', 'p1_atp_rank_points', 'p2_atp_rank', 'p2_atp_rank_points', 'p1_elo_rank',
                                'p1_elo_bestRank', 'p2_elo_rank', 'p2_elo_bestRank']

    features_to_have_z_score = ['minutes', 'p1_ace', 'p2_ace', 'p1_svpt', 'p2_svpt', 'p1_1stWon', 'p1_2ndWon',
                                'p2_1stWon', 'p2_2ndWon', 'p1_SvGms', 'p2_SvGms', 'p1_ht', 'p2_ht', 'p1_age', 'p2_age',
                                'p1_set1_score', 'p2_set1_score', 'set1_breakpoint_score', 'p1_set2_score',
                                'p2_set2_score', 'set2_breakpoint_score', 'p1_set3_score', 'p2_set3_score',
                                'set3_breakpoint_score', 'p1_set4_score', 'p2_set4_score', 'set4_breakpoint_score',
                                'p1_set5_score', 'p2_set5_score', 'set5_breakpoint_score']

    if not only_z_score:
        min_vals_dict = dict()
        max_vals_dict = dict()
        for feature in features_to_have_min_max:
            min_val = min_vals_dict[feature] = train[feature].min()
            max_val = max_vals_dict[feature] = train[feature].max()
            train[feature] = (train[feature] - min_val) / (max_val - min_val)
        print(train['draw_size'])

        print(f'Min-max normalize: all years, {min_vals_dict=}')
        print(f'Min-max normalize: all years, {max_vals_dict=}')

    # transform tourney_level to one-hot-encoding
    train['masters_1000s'] = np.where((train['tourney_level'] == 'M'), 1, -1)
    train['grand_slams'] = np.where((train['tourney_level'] == 'G'), 1, -1)
    train['other_tour-level'] = np.where((train['tourney_level'] == 'A'), 1, -1)
    train['challengers'] = np.where((train['tourney_level'] == 'C'), 1, -1)
    train['satellites_ITFs'] = np.where((train['tourney_level'] == 'S'), 1, -1)
    train['tour_finals'] = np.where((train['tourney_level'] == 'F'), 1, -1)
    train['davis_cup'] = np.where((train['tourney_level'] == 'D'), 1, -1)
    train = train.drop('tourney_level', axis='columns').copy()

    # transform pi_hand to binary number (R=1, L=-1)
    train['p1_hand'] = np.where((train['p1_hand'] == 'R'), 1, -1)
    train['p2_hand'] = np.where((train['p2_hand'] == 'R'), 1, -1)

    # run z-score normalize on numeric features
    mean_vals_dict = dict()
    std_vals_dict = dict()


    features_to_have_z_score = features_to_have_z_score + features_to_have_min_max if only_z_score else \
        features_to_have_z_score

    for feature in features_to_have_z_score:
        mean = mean_vals_dict[feature] = train[feature].mean()
        std = std_vals_dict[feature] = train[feature].std()
        train[feature] = (train[feature] - mean) / std

    print(f'Z-score normalize: all years, {mean_vals_dict=}')
    print(f'Z-score normalize: all years, {std_vals_dict=}')

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

    # Fill missing data
    features_with_missing_vals = {'p2_elo_bestRank', 'minutes', 'p1_atp_rank', 'p1_1stWon', 'p1_2ndWon', 'p2_svpt',
                                  'p2_bpFaced', 'p2_df', 'p2_ace', 'p2_1stWon', 'p1_ace', 'p2_atp_rank_points',
                                  'p1_svpt', 'p1_SvGms', 'p1_elo_bestRank', 'p1_atp_rank_points', 'p1_elo_rank',
                                  'p2_elo_rank', 'p1_bpSaved', 'p2_SvGms', 'p1_betting_odds', 'p2_2ndWon',
                                  'p2_bpSaved', 'p2_ht', 'p1_ht', 'p1_bpFaced', 'p1_df', 'p2_atp_rank',
                                  'p2_betting_odds', 'p1_age', 'p2_age'}

    features_mean = dict()
    for feature in features_with_missing_vals:
        features_mean[feature] = train[feature].mean()
    train = train.fillna(features_mean)
    print('Fill missimg data: all years features means:', features_mean)

    # Transform dates
    first_date = pd.to_datetime('01/01/2001')
    train['tourney_date'] = train['tourney_date'].apply(lambda x: pd.to_datetime(x))
    train['tourney_date'] = train['tourney_date'].apply(lambda x: (x - first_date).days)
    print(train['tourney_date'])
    train['tourney_date'] = train['tourney_date'] / 365
    print('**************************************')
    print(train['tourney_date'])
    all_years_path = 'Data/train/all_years_final_only_z_score.csv' if only_z_score else 'Data/train/all_years_final.csv'
    train.to_csv(all_years_path, index=False)


def plot_histograms_for_all_features():
    """
    Plot histogram bar for each feature for identify if it has standard distribution values.
    """
    data = pd.read_csv('Data/train/primary_all_years.csv')
    for feature in data.columns.values:
        try:
            sns.histplot(data[feature], bins=20, kde=True)
            plt.grid()
            plt.xlabel(feature)
            plt.show()
        except Exception as e:
            print(feature, e)


def prepare_test():
    """
    Prepare the test set for predict. Implement all data processing that was implemented on the train dataset.
    """
    normalize_betting_odds_features_for_test()
    drop_unused_features_test()
    remove_extra_chars_from_sets('test')
    all_years_tests = []
    for year in range(2001, 2022):
        all_years_tests.append(pd.read_csv(f'Data/test/more_dropped_data_{year}.csv'))

    test_all_years = pd.concat(all_years_tests)
    print(test_all_years.shape)
    print(test_all_years.head)
    test_all_years.to_csv('Data/test/primary_all_years.csv', index=False)
    transform_test_data(test_all_years)


def normalize_betting_odds_features_for_test():
    """
    normalize betting odds feature for each year, and do average of all feature to one.
    :return: save in new csv file
    """
    betting_features_by_year_means = {
        '2001': {'p1_CB': 1.8026585144927534, 'p2_CB': 1.7410416666666666, 'p1_GB': 1.7586625332152348, 'p2_GB': 1.7111647475642162, 'p1_IW': 1.6847927689594357, 'p2_IW': 1.6505820105820106, 'p1_SB': 1.5957377842283502, 'p2_SB': 1.5540909530720852},
        '2002': {'p1_B365': 1.214923932124049, 'p2_B365': 1.164800468110006, 'p1_CB': 1.9220489573889388, 'p2_CB': 1.878739800543971, 'p1_GB': 1.8801747311827959, 'p2_GB': 1.8424641577060932, 'p1_IW': 1.785685234305924, 'p2_IW': 1.7542484526967284, 'p1_SB': 1.2093738584474887, 'p2_SB': 1.2290017123287673},
        '2003': {'p1_B365': 1.8818850855745721, 'p2_B365': 1.8514572127139364, 'p1_CB': 1.6608628081457664, 'p2_CB': 1.6163504823151127, 'p1_IW': 1.7102326685660019, 'p2_IW': 1.698804368471035, 'p1_SB': 1.6742980516061086, 'p2_SB': 1.642156924697209, 'p1_B&W': 0.9207898089171974, 'p2_B&W': 0.9522738853503184},
        '2004': {'p1_B365': 1.738253802281369, 'p2_B365': 1.8406839353612168, 'p1_CB': 1.791569536423841, 'p2_CB': 1.87817880794702, 'p1_EX': 1.7788332546055738, 'p2_EX': 1.8461738308927726, 'p1_IW': 1.5780720037896732, 'p2_IW': 1.6169777356702986, 'p1_PS': 2.0035004686035616, 'p2_PS': 2.094783973758201},
        '2005': {'p1_B365': 2.0183264052905057, 'p2_B365': 2.052581955597544, 'p1_CB': 1.9831733966745844, 'p2_CB': 2.038014251781473, 'p1_EX': 1.9453546099290782, 'p2_EX': 1.995191489361702, 'p1_IW': 1.7796783216783216, 'p2_IW': 1.7971888111888112, 'p1_PS': 2.3769822843822843, 'p2_PS': 2.5030065268065265},
        '2006': {'p1_B365': 2.079852240228789, 'p2_B365': 2.096477597712107, 'p1_CB': 2.0543102625298326, 'p2_CB': 2.0867159904534605, 'p1_EX': 2.04096696984203, 'p2_EX': 2.063202489229296, 'p1_PS': 2.5899727827311123, 'p2_PS': 2.5689080244016895, 'p1_UB': 2.0995336538461538, 'p2_UB': 2.143889423076923},
        '2007': {'p1_B365': 2.041780341023062, 'p2_B365': 2.202387161484445, 'p1_CB': 1.9914708886618937, 'p2_CB': 2.1476046986721093, 'p1_EX': 1.934487704918035, 'p2_EX': 2.0357479508196743, 'p1_PS': 2.463482053838486, 'p2_PS': 2.9621899302093726, 'p1_UB': 2.0487780040733186, 'p2_UB': 2.28132382892057},
        '2008': {'p1_B365': 2.1108969276511402, 'p2_B365': 2.1794103072348863, 'p1_EX': 1.9833924623115577, 'p2_EX': 2.0470035175879397, 'p1_LB': 1.9660845350571856, 'p2_LB': 2.0275768274490304, 'p1_PS': 2.4444945490584735, 'p2_PS': 2.5674876114965315, 'p1_UB': 2.0689253731343284, 'p2_UB': 2.1496915422885574},
        '2009': {'p1_B365': 2.2757308467741932, 'p2_B365': 2.3472575604838712, 'p1_EX': 2.1268405063291143, 'p2_EX': 2.2215898734177215, 'p1_LB': 2.1063171225937185, 'p2_LB': 2.1763850050658564, 'p1_SJ': 2.1190066428206435, 'p2_SJ': 2.163454777721001, 'p1_UB': 2.216681841173495, 'p2_UB': 2.3044764795144164},
        '2010': {'p1_B365': 2.2164541772151907, 'p2_B365': 2.2630627212948915, 'p1_EX': 2.059028484231943, 'p2_EX': 2.099684638860631, 'p1_LB': 2.10080476673428, 'p2_LB': 2.156466531440162, 'p1_PS': 2.5171191079574253, 'p2_PS': 2.655035985808414, 'p1_SJ': 2.0648633350331465, 'p2_SJ': 2.112802651708312, 'p1_Max': 2.0262075134168156, 'p2_Max': 2.0297912939773406, 'p1_Avg': 1.6439355992844367, 'p2_Avg': 1.634102564102564},
        '2011': {'p1_B365': 2.4721627558662007, 'p2_B365': 2.2667716849451645, 'p1_EX': 2.2511738261738263, 'p2_EX': 2.081613386613386, 'p1_LB': 2.2647326336831584, 'p2_LB': 2.0622188905547225, 'p1_PS': 2.9789865202196704, 'p2_PS': 2.5765401897154265, 'p1_SJ': 2.329057644110276, 'p2_SJ': 2.0806713426853705, 'p1_Max': 3.2758594917787747, 'p2_Max': 2.7502341803687096, 'p1_Avg': 2.4178674638764326, 'p2_Avg': 2.20293971101146},
        '2012': {'p1_B365': 2.500715985514744, 'p2_B365': 2.6768599483204136, 'p1_EX': 2.1558298538622127, 'p2_EX': 2.240067849686848, 'p1_LB': 2.229632124352332, 'p2_LB': 2.3486542443064184, 'p1_PS': 2.858189342990171, 'p2_PS': 3.1277496120020696, 'p1_SJ': 2.2381606217616583, 'p2_SJ': 2.3110973084886126, 'p1_Max': 3.053109504132231, 'p2_Max': 3.397463842975207, 'p1_Avg': 2.3170919421487604, 'p2_Avg': 2.440222107438016},
        '2013': {'p1_B365': 2.5642945169712794, 'p2_B365': 2.4445932114882507, 'p1_EX': 2.21980607966457, 'p2_EX': 2.125047169811321, 'p1_LB': 2.2805293501048216, 'p2_LB': 2.193045073375262, 'p1_PS': 2.696614338042909, 'p2_PS': 2.617247514390371, 'p1_SJ': 2.2961627296587928, 'p2_SJ': 2.1830866141732286, 'p1_Max': 2.989759916492693, 'p2_Max': 2.880240083507307, 'p1_Avg': 2.3858611691022964, 'p2_Avg': 2.280135699373695},
        '2014': {'p1_B365': 2.4730547368421054, 'p2_B365': 2.448013144058885, 'p1_EX': 2.242382956338769, 'p2_EX': 2.173277222514466, 'p1_LB': 2.2939440337909183, 'p2_LB': 2.2341341077085533, 'p1_PS': 2.6155970541820093, 'p2_PS': 2.574392425039453, 'p1_SJ': 2.2947665245202558, 'p2_SJ': 2.2039498933901918, 'p1_Max': 2.8648371848739496, 'p2_Max': 2.809227941176471, 'p1_Avg': 2.3657720588235294, 'p2_Avg': 2.297578781512605},
        '2015': {'p1_B365': 2.4818091523660946, 'p2_B365': 2.5086645865834636, 'p1_EX': 2.2250130005200206, 'p2_EX': 2.2389391575663025, 'p1_LB': 2.2549817992719707, 'p2_LB': 2.3320072802912115, 'p1_PS': 2.6270930837233486, 'p2_PS': 2.710228809152366, 'p1_Max': 24.674919480519478, 'p2_Max': 2.910768831168831, 'p1_Avg': 2.339220779220779, 'p2_Avg': 2.3919792207792203},
        '2016': {'p1_B365': 2.489977914740626, 'p2_B365': 2.379774011299435, 'p1_EX': 2.1506214689265537, 'p2_EX': 2.063287108371854, 'p1_LB': 2.381406491499227, 'p2_LB': 2.28069036579083, 'p1_PS': 2.720683102208526, 'p2_PS': 2.5716076014381097, 'p1_Max': 24.733461736004106, 'p2_Max': 2.671931176168464, 'p1_Avg': 2.3971443246019515, 'p2_Avg': 2.287904468412943},
        '2017': {'p1_B365': 2.4181681370005186, 'p2_B365': 2.293832381940841, 'p1_EX': 2.1719490644490644, 'p2_EX': 2.0939760914760916, 'p1_LB': 2.2404508419337317, 'p2_LB': 2.115708853883759, 'p1_PS': 2.60496878251821, 'p2_PS': 2.473511966701353, 'p1_Max': 2.7489885892116184, 'p2_Max': 2.582349585062241, 'p1_Avg': 2.3761099585062238, 'p2_Avg': 2.271260373443983},
        '2018': {'p1_B365': 2.1183897435897436, 'p2_B365': 2.263584615384615, 'p1_EX': 2.0104671457905545, 'p2_EX': 2.133372689938398, 'p1_LB': 1.8250674915635547, 'p2_LB': 1.9405736782902139, 'p1_PS': 2.258998459167951, 'p2_PS': 2.4226502311248073, 'p1_Max': 2.389334698055271, 'p2_Max': 2.5685568065506654, 'p1_Avg': 2.1269089048106444, 'p2_Avg': 2.2719089048106444},
        '2019': {'p1_B365': 2.1239379203310915, 'p2_B365': 2.293970512157269, 'p1_PS': 2.3104503105590064, 'p2_PS': 2.4865036231884057, 'p1_Max': 2.4296642561983472, 'p2_Max': 2.637933884297521, 'p1_Avg': 2.15323347107438, 'p2_Avg': 2.278884297520661},
        '2020': {'p1_B365': 2.3243584337349397, 'p2_B365': 2.2497469879518066, 'p1_PS': 2.4717965760322254, 'p2_PS': 2.4821802618328297, 'p1_Max': 2.6668937875751504, 'p2_Max': 2.6592384769539077, 'p1_Avg': 2.260571142284569, 'p2_Avg': 2.25562124248497},
        '2021': {'p1_B365': 2.2139303944315545, 'p2_B365': 2.282766434648105, 'p1_PS': 2.3412838360402164, 'p2_PS': 2.385197215777262, 'p1_Max': 2.5070765661252903, 'p2_Max': 2.6487703016241295, 'p1_Avg': 2.1956535189481827, 'p2_Avg': 2.196426914153132}}

    betting_features_by_year_std = {
        '2001': {'p1_CB': 1.2986783729317912, 'p2_CB': 1.2406706322211158, 'p1_GB': 1.0897659345638016, 'p2_GB': 1.0424555289220723, 'p1_IW': 0.9916855502252442, 'p2_IW': 0.9592806799440907, 'p1_SB': 1.3349712974060137, 'p2_SB': 1.250189792542305},
        '2002': {'p1_B365': 1.4597651510835634, 'p2_B365': 1.3954147929399892, 'p1_CB': 1.286760762607844, 'p2_CB': 1.2437555404032092, 'p1_GB': 1.1705462396943533, 'p2_GB': 1.1048209517089653, 'p1_IW': 0.9763051432884652, 'p2_IW': 0.9426720264640949, 'p1_SB': 1.4260563858431632, 'p2_SB': 1.4335206498171678},
        '2003': {'p1_B365': 1.5930276206009764, 'p2_B365': 1.539174864006456, 'p1_CB': 1.6976467091231677, 'p2_CB': 1.622483557302898, 'p1_IW': 1.1458247630614582, 'p2_IW': 1.140813768268793, 'p1_SB': 1.6192450284700164, 'p2_SB': 1.5737551755596753, 'p1_B&W': 1.294595034666788, 'p2_B&W': 1.3481158823589738},
        '2004': {'p1_B365': 1.53556718522731, 'p2_B365': 1.6506125520150747, 'p1_CB': 1.5737963247758482, 'p2_CB': 1.6142400051807309, 'p1_EX': 1.4953887246076782, 'p2_EX': 1.5395371005423137, 'p1_IW': 1.1347964414525524, 'p2_IW': 1.1586006263386905, 'p1_PS': 2.3628583092004254, 'p2_PS': 2.2129926899478747},
        '2005': {'p1_B365': 1.897308589917626, 'p2_B365': 1.9734013971341733, 'p1_CB': 1.8587759936929433, 'p2_CB': 1.954203656084643, 'p1_EX': 1.80655552076165, 'p2_EX': 1.8958592218181916, 'p1_IW': 1.2823315471624053, 'p2_IW': 1.3120843955587544, 'p1_PS': 3.3767100262481993, 'p2_PS': 3.715308414227615},
        '2006': {'p1_B365': 2.0632727513454614, 'p2_B365': 2.06244592645373, 'p1_CB': 1.978950280132515, 'p2_CB': 2.0379364113302825, 'p1_EX': 2.0486411072612483, 'p2_EX': 2.0125279531324267, 'p1_PS': 4.442913060071373, 'p2_PS': 4.120712246879593, 'p1_UB': 2.530310401904795, 'p2_UB': 2.802716235127126},
        '2007': {'p1_B365': 2.180736787227609, 'p2_B365': 2.4863723745858057, 'p1_CB': 2.04371716454765, 'p2_CB': 2.4921016228760386, 'p1_EX': 1.8616773034596386, 'p2_EX': 2.1386769417072147, 'p1_PS': 4.185945991150019, 'p2_PS': 6.303058369655946, 'p1_UB': 2.7506991346933134, 'p2_UB': 3.4897324659163256},
        '2008': {'p1_B365': 2.423486036568033, 'p2_B365': 2.449929227905971, 'p1_EX': 2.032268640285543, 'p2_EX': 2.0703565829539805, 'p1_LB': 1.9824871393093288, 'p2_LB': 2.0208185158667096, 'p1_PS': 4.213474321675045, 'p2_PS': 4.138138672504306, 'p1_UB': 2.514320248298428, 'p2_UB': 2.5422151451204424},
        '2009': {'p1_B365': 3.120191620881602, 'p2_B365': 2.8537661414009823, 'p1_EX': 2.449491054467573, 'p2_EX': 2.391945303355389, 'p1_LB': 2.529600574826152, 'p2_LB': 2.3351474537409187, 'p1_SJ': 2.6298087401310295, 'p2_SJ': 2.423292197246359, 'p1_UB': 3.014743111574262, 'p2_UB': 2.833887803454642},
        '2010': {'p1_B365': 2.4815488632486153, 'p2_B365': 2.6684726901522415, 'p1_EX': 2.056870748128404, 'p2_EX': 2.2395890667773286, 'p1_LB': 2.192823698249789, 'p2_LB': 2.4609682474098586, 'p1_PS': 3.6817667791606032, 'p2_PS': 4.742478394974264, 'p1_SJ': 2.1863953837698937, 'p2_SJ': 2.375100721669419, 'p1_Max': 3.820764246466871, 'p2_Max': 3.997412948221826, 'p1_Avg': 2.231389330506615, 'p2_Avg': 2.3239145301948905},
        '2011': {'p1_B365': 3.23196174571189, 'p2_B365': 2.7104397496761092, 'p1_EX': 2.5683797778779796, 'p2_EX': 2.0673906482168416, 'p1_LB': 2.7292449580386715, 'p2_LB': 2.0115394786240137, 'p1_PS': 5.887365399732554, 'p2_PS': 4.162745549663841, 'p1_SJ': 3.3081779787069903, 'p2_SJ': 2.1669128335655494, 'p1_Max': 6.9724230434546675, 'p2_Max': 4.728908781123511, 'p1_Avg': 3.082609413141847, 'p2_Avg': 2.4663274346298976},
        '2012': {'p1_B365': 3.9273331961629196, 'p2_B365': 4.476417206626269, 'p1_EX': 2.4680226205670146, 'p2_EX': 2.7037981882739692, 'p1_LB': 2.877470236249537, 'p2_LB': 3.369195028927437, 'p1_PS': 5.913857986546655, 'p2_PS': 7.140291404487643, 'p1_SJ': 3.180530958002038, 'p2_SJ': 3.3852116835995463, 'p1_Max': 6.307244351370834, 'p2_Max': 7.959760181273983, 'p1_Avg': 3.015115368624543, 'p2_Avg': 3.449231319414347},
        '2013': {'p1_B365': 3.65599592901355, 'p2_B365': 3.556971742664652, 'p1_EX': 2.346859020435263, 'p2_EX': 2.2678738400668887, 'p1_LB': 2.614075506307672, 'p2_LB': 2.601640139383709, 'p1_PS': 4.045662986766926, 'p2_PS': 4.364690505087009, 'p1_SJ': 2.5637595131275885, 'p2_SJ': 2.4022310020347546, 'p1_Max': 5.09439908952194, 'p2_Max': 5.299756656789157, 'p1_Avg': 2.8365492844047164, 'p2_Avg': 2.801367119130453},
        '2014': {'p1_B365': 3.298162438451047, 'p2_B365': 3.6838219558431606, 'p1_EX': 2.162731205189132, 'p2_EX': 2.2024667131607036, 'p1_LB': 2.6145453433410686, 'p2_LB': 2.7995496892757696, 'p1_PS': 3.9380890134111, 'p2_PS': 3.855127236841519, 'p1_SJ': 2.477261557094876, 'p2_SJ': 2.469669498135595, 'p1_Max': 4.829924791234243, 'p2_Max': 5.004837426880907, 'p1_Avg': 2.663743278320462, 'p2_Avg': 2.749777933821181},
        '2015': {'p1_B365': 3.8626514901448017, 'p2_B365': 3.6300399736835187, 'p1_EX': 2.254688198709546, 'p2_EX': 2.2925363263055414, 'p1_LB': 2.5461376836783836, 'p2_LB': 2.9775020828595435, 'p1_PS': 3.669431991414196, 'p2_PS': 4.147095739112772, 'p1_Max': 960.315942792123, 'p2_Max': 5.012927978117072, 'p1_Avg': 2.6636609628741654, 'p2_Avg': 2.9302818366122465},
        '2016': {'p1_B365': 3.3817910450194675, 'p2_B365': 3.2742513482243396, 'p1_EX': 2.0197938988024418, 'p2_EX': 1.9821093390413438, 'p1_LB': 2.763544830660657, 'p2_LB': 2.725111846272698, 'p1_PS': 4.328822378157985, 'p2_PS': 3.7175421814164027, 'p1_Max': 965.0732268228942, 'p2_Max': 4.009450227886112, 'p1_Avg': 2.750334028339715, 'p2_Avg': 2.5529677277688716},
        '2017': {'p1_B365': 2.754875677165376, 'p2_B365': 2.4835863022024682, 'p1_EX': 1.8472252372050906, 'p2_EX': 1.76313678345181, 'p1_LB': 2.4339585003474675, 'p2_LB': 2.224930885583337, 'p1_PS': 3.10878082094478, 'p2_PS': 2.85093347035343, 'p1_Max': 3.748458498443829, 'p2_Max': 3.1888732066621848, 'p1_Avg': 2.4424454368165627, 'p2_Avg': 2.268828551935126},
        '2018': {'p1_B365': 2.334204966130238, 'p2_B365': 2.5671343868710403, 'p1_EX': 1.7503919987842655, 'p2_EX': 1.9045960874975818, 'p1_LB': 1.8998934072373173, 'p2_LB': 2.27275668922143, 'p1_PS': 2.58361816245167, 'p2_PS': 2.8710347000916263, 'p1_Max': 2.988419596054523, 'p2_Max': 3.2247350264607766, 'p1_Avg': 2.0900285449931917, 'p2_Avg': 2.3711099740101713},
        '2019': {'p1_B365': 2.104595742515731, 'p2_B365': 2.8655130224159153, 'p1_PS': 2.525089638848592, 'p2_PS': 3.26913774226871, 'p1_Max': 2.868258263931822, 'p2_Max': 3.8675050638470103, 'p1_Avg': 2.00457151413518, 'p2_Avg': 2.444262721818726},
        '2020': {'p1_B365': 3.2446850751999468, 'p2_B365': 3.2279815427668077, 'p1_PS': 3.4952478122532087, 'p2_PS': 3.5533314253642287, 'p1_Max': 4.193779647757297, 'p2_Max': 4.246221262047147, 'p1_Avg': 2.68794264847548, 'p2_Avg': 2.7134869816582032},
        '2021': {'p1_B365': 2.368770696542486, 'p2_B365': 3.310031774919654, 'p1_PS': 2.596947074740498, 'p2_PS': 3.4029145533212946, 'p1_Max': 2.9674126064554, 'p2_Max': 4.496964950420369, 'p1_Avg': 2.1757354140456204, 'p2_Avg': 2.6629144820976642}}

    betting_features_by_year_by_player = dict()
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/p1p2_after_mismatch{year}.csv')
        train, test = train_test_split(data, train_size=0.8, random_state=11)
        all_features = test.columns.values
        betting_features_by_year_by_player[f'{year}_p1'] = []
        betting_features_by_year_by_player[f'{year}_p2'] = []
        for feature in all_features:
            if feature in BETTING_HOUSES:
                if 'p1' in feature:
                    betting_features_by_year_by_player[f'{year}_p1'].append(feature)
                else:
                    betting_features_by_year_by_player[f'{year}_p2'].append(feature)

                test[feature] = (test[feature] - betting_features_by_year_means[str(year)][feature]) / betting_features_by_year_std[str(year)][feature]

        test['p1_betting_odds'] = test[betting_features_by_year_by_player[f'{year}_p1']].mean(axis=1)
        test['p2_betting_odds'] = test[betting_features_by_year_by_player[f'{year}_p2']].mean(axis=1)

        test_with_avg_betting_odds_only = test.drop(betting_features_by_year_by_player[f'{year}_p1'],
                                                      axis='columns').copy()
        test_with_avg_betting_odds_only = test_with_avg_betting_odds_only.drop(
            betting_features_by_year_by_player[f'{year}_p2'], axis='columns').copy()

        test_with_avg_betting_odds_only.to_csv(f'Data/test/only_avg_betting_odds_{year}.csv', index=False)


def drop_unused_features_test():
    """
    Drop chosen features from test data (same features where dropped from the train)
    :return: save in new csv file
    """
    for year in range(2001, 2022):
        test = pd.read_csv(f'Data/test/only_avg_betting_odds_{year}.csv')
        print(f'{year} Before drop, {len(test.columns.values)}')

        candidates = ['p1_Pts', 'p2_Pts', 'p1_elo_points', 'p2_elo_points', 'p2_elo_bestPoints', 'tourney_name',
                      'p1_elo_bestPoints', 'p1_1stIn', 'p2_1stIn', 'tourney_id', 'match_num', 'Court']
        test = test.drop([x for x in candidates if x in test.columns], axis='columns').copy()

        test.to_csv(f'Data/test/more_dropped_data_{year}.csv', index=False)

        print(f'{year} After drop, {len(test.columns.values)}')


def transform_test_data(test: pd.DataFrame):
    """
    Transform values of test set: min-max, z-score, one-hot-encodeing, fill missing vales and transform dates.
    :param test: the current test set
    :return: save the processed table in new csv file.
    """
    test = min_max_normalize_test(test)
    test = z_score_normalaize_test(test)
    test = apply_one_hot_encoding(test)
    test = fill_missing_test_data(test)
    test = transform_tests_dates(test)
    test.to_csv('Data/test/test_all_years_final.csv', index=False)


def min_max_normalize_test(test: pd.DataFrame):
    """
    Implement min-max normalization on the test set. The min, max values of each feature were printed when the min-max
    on the train set was calculated, and here the dicts are copies of it.
    :param test: the current test set
    :return: the test table after normalization
    """
    min_vals_dict = {'draw_size': 4, 'p1_df': 0.0, 'p1_bpSaved': 0.0, 'p1_bpFaced': 0.0, 'p2_df': 0.0,
                     'p2_bpSaved': 0.0, 'p2_bpFaced': 0.0, 'p1_atp_rank': 0.0, 'p1_atp_rank_points': 0.0,
                     'p2_atp_rank': 0.0, 'p2_atp_rank_points': 0.0, 'p1_elo_rank': 0.0, 'p1_elo_bestRank': 0.0,
                     'p2_elo_rank': 0.0, 'p2_elo_bestRank': 0.0}
    max_vals_dict = {'draw_size': 128, 'p1_df': 21.0, 'p1_bpSaved': 25.0, 'p1_bpFaced': 31.0, 'p2_df': 26.0,
                     'p2_bpSaved': 24.0, 'p2_bpFaced': 31.0, 'p1_atp_rank': 2147.0, 'p1_atp_rank_points': 16790.0,
                     'p2_atp_rank': 2159.0, 'p2_atp_rank_points': 16950.0, 'p1_elo_rank': 236.0,
                     'p1_elo_bestRank': 231.0, 'p2_elo_rank': 236.0, 'p2_elo_bestRank': 231.0}

    for feature in min_vals_dict.keys():
        test[feature] = (test[feature] - min_vals_dict[feature]) / (max_vals_dict[feature] - min_vals_dict[feature])

    return test


def z_score_normalaize_test(test: pd.DataFrame):
    """
    Implement z-score normalization on the test set. The mean, std values of each feature were printed when the z-score
    on the train set was calculated, and here the dicts are copies of it.
    :param test: the current test set
    :return: the test table after normalization
    """
    mean_vals_dict = {'minutes': 107.46406599882145, 'p1_ace': 5.846042056589233, 'p2_ace': 5.832896620059452,
                      'p1_svpt': 76.89981283716834, 'p2_svpt': 76.94105471760432, 'p1_1stWon': 33.23939227127601,
                      'p1_2ndWon': 15.300605526808324, 'p2_1stWon': 33.23749862380271, 'p2_2ndWon': 15.315424419244742,
                      'p1_SvGms': 12.012969283276451, 'p2_SvGms': 12.011449961466475, 'p1_ht': 176.13105178903646,
                      'p2_ht': 180.77752878791242, 'p1_age': 26.406553007601488, 'p2_age': 26.385000075875336,
                      'p1_set1_score': 4.928161704452336, 'p2_set1_score': 4.928434853865064,
                      'set1_breakpoint_score': 0.7902422625175971, 'p1_set2_score': 4.863005063769882,
                      'p2_set2_score': 4.857416006555586, 'set2_breakpoint_score': 0.7189922887819638,
                      'p1_set3_score': 2.417645452062278, 'p2_set3_score': 2.4005841195133737,
                      'set3_breakpoint_score': 0.34549198411531107, 'p1_set4_score': 0.5551026411447062,
                      'p2_set4_score': 0.5417813544008573, 'set4_breakpoint_score': 0.08295337549639653,
                      'p1_set5_score': 0.22847897800096653, 'p2_set5_score': 0.2236253230517093,
                      'set5_breakpoint_score': 0.007837286995986804}
    std_vals_dict = {'minutes': 40.56282956466988, 'p1_ace': 5.431342662017276, 'p2_ace': 5.356512692178133,
                     'p1_svpt': 32.52161767308845, 'p2_svpt': 32.51286021344843, 'p1_1stWon': 15.368943214731674,
                     'p1_2ndWon': 7.657772465478649, 'p2_1stWon': 15.3611249427521, 'p2_2ndWon': 7.625312169236094,
                     'p1_SvGms': 4.787572175942046, 'p2_SvGms': 4.785558819589335, 'p1_ht': 41.81385521065151,
                     'p2_ht': 30.92808792364547, 'p1_age': 3.992881470164066, 'p2_age': 3.983837322891673,
                     'p1_set1_score': 1.7897291583699015, 'p2_set1_score': 1.8002461094629774,
                     'set1_breakpoint_score': 1.8906202800736172, 'p1_set2_score': 1.8296092523543466,
                     'p2_set2_score': 1.8416833241334194, 'set2_breakpoint_score': 1.8210621901005168,
                     'p1_set3_score': 2.7602156004826965, 'p2_set3_score': 2.7464576420895974,
                     'set3_breakpoint_score': 1.3141112791734693, 'p1_set4_score': 1.6704680760542685,
                     'p2_set4_score': 1.6460727659690937, 'set4_breakpoint_score': 0.6594175352751057,
                     'p1_set5_score': 1.2267635152431997, 'p2_set5_score': 1.2120257691543308,
                     'set5_breakpoint_score': 0.20683788496359456}

    for feature in mean_vals_dict.keys():
        test[feature] = (test[feature] - mean_vals_dict[feature]) / std_vals_dict[feature]

    return test


def apply_one_hot_encoding(test):
    """
    Implement one-hot-encoding for categorical features
    :param test: the current test set
    :return: the test table after apply one-hot-encoding
    """
    # transform surface to one-hot-encoding
    test['carpet'] = np.where((test['surface'] == 'Carpet'), 1, -1)
    test['clay'] = np.where((test['surface'] == 'Clay'), 1, -1)
    test['grass'] = np.where((test['surface'] == 'Grass'), 1, -1)
    test['hard'] = np.where((test['surface'] == 'Hard'), 1, -1)
    test = test.drop('surface', axis='columns').copy()

    # transform tourney_level to one-hot-encoding
    test['masters_1000s'] = np.where((test['tourney_level'] == 'M'), 1, -1)
    test['grand_slams'] = np.where((test['tourney_level'] == 'G'), 1, -1)
    test['other_tour-level'] = np.where((test['tourney_level'] == 'A'), 1, -1)
    test['challengers'] = np.where((test['tourney_level'] == 'C'), 1, -1)
    test['satellites_ITFs'] = np.where((test['tourney_level'] == 'S'), 1, -1)
    test['tour_finals'] = np.where((test['tourney_level'] == 'F'), 1, -1)
    test['davis_cup'] = np.where((test['tourney_level'] == 'D'), 1, -1)
    test = test.drop('tourney_level', axis='columns').copy()

    # transform pi_hand to binary number (R=1, L=-1)
    test['p1_hand'] = np.where((test['p1_hand'] == 'R'), 1, -1)
    test['p2_hand'] = np.where((test['p2_hand'] == 'R'), 1, -1)

    # transform best_of to binary number (3=1, 5=-1)
    test['best_of'] = np.where((test['best_of'] == 3), 1, -1)

    # transform round to one-hot-encoding
    test['f_round'] = np.where((test['round'] == 'F'), 1, -1)
    test['qf_round'] = np.where((test['round'] == 'QF'), 1, -1)
    test['r128_round'] = np.where((test['round'] == 'R128'), 1, -1)
    test['r16_round'] = np.where((test['round'] == 'R16'), 1, -1)
    test['r32_round'] = np.where((test['round'] == 'R32'), 1, -1)
    test['r64_round'] = np.where((test['round'] == 'R64'), 1, -1)
    test['rr_round'] = np.where((test['round'] == 'RR'), 1, -1)
    test['sf_round'] = np.where((test['round'] == 'SF'), 1, -1)
    test = test.drop('round', axis='columns').copy()

    # transform p1_won to signed 1/-1 instead of unsigned 1/0
    test['p1_won'] = np.where((test['p1_won'] == 1), 1, -1)

    return test


def fill_missing_test_data(test):
    """
    Fiil missing values on test set by the values that were printed when the means to fill missing values on train were
    calculated.
    :param test: the current test set
    :return: the test table after apply one-hot-encoding
    """
    means = {'p2_betting_odds': 0.013108076564071716, 'p2_atp_rank': 0.045668815063624385,
            'p2_2ndWon': 1.0013153162753806e-16, 'p2_elo_rank': 0.27468119002281255, 'minutes': -1.0718853291372165e-16,
            'p2_age': 2.0068707299812434e-16, 'p2_df': 0.11167946883018996, 'p2_ace': 3.003945948826142e-17,
            'p1_elo_rank': 0.26629257885810864, 'p2_elo_bestRank': 0.14148187741068718,
            'p2_bpSaved': 0.16625289001431245, 'p2_atp_rank_points': 0.07550748411717913,
            'p1_svpt': -1.251644145344226e-16, 'p2_SvGms': 1.451907208599302e-16, 'p1_elo_bestRank': 0.1365559144746962,
            'p2_bpFaced': 0.21315822184655492, 'p1_SvGms': -1.6521702718543781e-16, 'p1_bpFaced': 0.2121396582768944,
            'p1_betting_odds': -0.00103945163478074, 'p1_atp_rank_points': 0.07481398897918069, 'p1_1stWon': 0.0,
            'p1_ht': 2.262620402022385e-16, 'p2_svpt': -2.5032882906884516e-17, 'p1_df': 0.1376459424461186,
            'p1_atp_rank': 0.046167961549591845, 'p2_ht': -3.7331437027798743e-16, 'p1_ace': -1.2516441453442258e-17,
            'p1_age': -8.122194752122179e-17, 'p1_2ndWon': -1.1890619380770144e-16, 'p1_bpSaved': 0.1585476164262909,
            'p2_1stWon': 4.0052612651015227e-17}

    test = test.fillna(means)
    return test


def transform_tests_dates(test):
    """
    Transform dates of 'tourney_date' feature by counting days from 01/01/2001 to the specific date and divide by 365.
    :param test: the current test set
    :return: the test table after transform the dates
    """
    first_date = pd.to_datetime('01/01/2001')
    test['tourney_date'] = test['tourney_date'].apply(lambda x: pd.to_datetime(x))
    test['tourney_date'] = test['tourney_date'].apply(lambda x: (x - first_date).days)
    print(test['tourney_date'])
    test['tourney_date'] = test['tourney_date'] / 365
    print('**************************************')
    print(test['tourney_date'])

    return test


def normalize_player_id_test():
    """
    Normalize the player ID features for reduse them to the same scale of the other features.
    :return: save the table in new csv file
    """
    test_all_years = pd.read_csv(f'Data/test/test_all_years_final.csv')

    MIN_PLAYER_ID = 100644
    MAX_PLAYER_ID = 210013

    test_all_years['p1_id'] = (test_all_years['p1_id'] - 100644) / (210013 - 100644)
    test_all_years['p2_id'] = (test_all_years['p2_id'] - 100644) / (210013 - 100644)

    test_all_years.to_csv('Data/test/test_all_years_final.csv', index=False)


def create_three_train_tables():
    """
    Create 3 different files, one for each question. These tables will be use for the training and tuning the models.
    Q1 - all features, no changes
    Q2 - all features exclude score set features
    Q3 - only static data features
    :return: Save all 3 tables to csv files
    """
    all_years_train = pd.read_csv('Data/train/all_years_final.csv')

    MIN_PLAYER_ID = 100644
    MAX_PLAYER_ID = 210013

    all_years_train['p1_id'] = (all_years_train['p1_id'] - 100644) / (210013 - 100644)
    all_years_train['p2_id'] = (all_years_train['p2_id'] - 100644) / (210013 - 100644)
    all_years_train.to_csv('Data/train/all_years_final_normal_id.csv', index=False)

    train_without_scores = all_years_train.drop([x for x in all_years_train.columns.values if 'set' in x], axis=1)
    train_without_scores.to_csv('Data/train/train_without_scores.csv', index=False)
    static_feature_match = ['draw_size', 'tourney_date', 'p1_id', 'p1_hand', 'p1_ht', 'p1_age', 'p2_id', 'p2_hand',
                            'p2_ht', 'p2_age', 'best_of', 'p1_atp_rank', 'p1_atp_rank_points', 'p2_atp_rank',
                            'p2_atp_rank_points', 'p1_elo_rank', 'p1_elo_bestRank', 'p2_elo_rank', 'p2_elo_bestRank',
                            'p1_won', 'carpet', 'clay', 'grass', 'hard', 'masters_1000s', 'grand_slams',
                            'other_tour-level', 'challengers', 'satellites_ITFs', 'tour_finals', 'davis_cup', 'f_round',
                            'qf_round', 'r128_round', 'r16_round', 'r32_round', 'r64_round', 'rr_round', 'sf_round']
    train_static_match_data = all_years_train[static_feature_match]
    train_static_match_data.to_csv('Data/train/train_static_match_data.csv', index=False)


def create_three_test_tables():
    """
    Create 3 different files, one for each question. These tables will be use for testing the tuned models.
    Q1 - all features, no changes
    Q2 - all features exclude score set features
    Q3 - only static data features
    :return: Save all 3 tables to csv files
    """
    all_years_test = pd.read_csv('Data/test/test_all_years_final.csv')

    test_without_scores = all_years_test.drop([x for x in all_years_test.columns.values if 'set' in x], axis=1)
    test_without_scores.to_csv('Data/test/test_without_scores.csv', index=False)
    static_feature_match = ['draw_size', 'tourney_date', 'p1_id', 'p1_hand', 'p1_ht', 'p1_age', 'p2_id', 'p2_hand',
                            'p2_ht', 'p2_age', 'best_of', 'p1_atp_rank', 'p1_atp_rank_points', 'p2_atp_rank',
                            'p2_atp_rank_points', 'p1_elo_rank', 'p1_elo_bestRank', 'p2_elo_rank', 'p2_elo_bestRank',
                            'p1_won', 'carpet', 'clay', 'grass', 'hard', 'masters_1000s', 'grand_slams',
                            'other_tour-level', 'challengers', 'satellites_ITFs', 'tour_finals', 'davis_cup', 'f_round',
                            'qf_round', 'r128_round', 'r16_round', 'r32_round', 'r64_round', 'rr_round', 'sf_round']
    test_static_match_data = all_years_test[static_feature_match]
    test_static_match_data.to_csv('Data/test/test_static_match_data.csv', index=False)


def main():
    #print_all_features()  # mainly for debug
    #find_correlation()
    normalize_betting_odds_features()
    drop_unsued_features()
    remove_extra_chars_from_sets()
    create_all_years_uniform_data()
    plot_histograms_for_all_features()
    prepare_all_years_data(only_z_score=False)
    prepare_test()
    normalize_player_id_test()
    create_three_train_tables()
    create_three_test_tables()


if __name__ == '__main__':
    main()
