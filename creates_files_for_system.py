import pandas as pd
import numpy as np
import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier


TARGET = 'p1_won'


def create_players_table():
    """
    Create table of players full name and their player ID for easy match between name to ID. the table also contain date
    of birth of the players.
    :return: save the new table in files folder. (the folder contains all necessary files for the system)
    """
    players_data = pd.read_csv('Data/relevant_players.csv')
    players_data['full_name'] = players_data['first_name'] + ' ' + players_data['last_name']
    players_data.drop(labels=['first_name', 'last_name', 'country'], axis=1, inplace=True)
    players_data.to_csv('files/players_data.csv', index=False)


def create_ranking_table():
    """
    Create table of elo ranking of players by each year, for easy finding of player's elo ranking in a specific year.
    :return: save the new table in files folder. (the folder contains all necessary files for the system)
    """
    atp_ranks_00s = pd.read_csv('Data/atp_rankings_00s.csv')
    atp_ranks_10s = pd.read_csv('Data/atp_rankings_10s.csv')

    atp_ranks_00s['ranking_date'] = atp_ranks_00s['ranking_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    atp_ranks_10s['ranking_date'] = atp_ranks_10s['ranking_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

    atp_ranks_00s['year'] = atp_ranks_00s['ranking_date'].dt.year
    atp_ranks_10s['year'] = atp_ranks_10s['ranking_date'].dt.year

    atp_ranks_00s['elo_rank'] = np.nan
    atp_ranks_00s['elo_best_rank'] = np.nan
    atp_ranks_10s['elo_rank'] = np.nan
    atp_ranks_10s['elo_best_rank'] = np.nan

    for year in range(2001, 2010):
        elo_ranks = pd.read_csv(f'Data/elo_ranking/{year}.csv')
        elo_ranks.rename(columns={'rank': 'elo_rank'}, inplace=True)

        for index, row in elo_ranks.iterrows():
            player_id = row.player_id
            elo_rank_this_year = row.elo_rank
            best_elo_rank = row.bestRank

            atp_ranks_00s.loc[(atp_ranks_00s['player'] == player_id) & (atp_ranks_00s['year'] == year), ['elo_rank']] = elo_rank_this_year
            atp_ranks_00s.loc[(atp_ranks_00s['player'] == player_id) & (atp_ranks_00s['year'] == year), ['elo_best_rank']] = best_elo_rank

    for year in range(2011, 2022):
        elo_ranks = pd.read_csv(f'Data/elo_ranking/{year}.csv')
        elo_ranks.rename(columns={'rank': 'elo_rank'}, inplace=True)

        for index, row in elo_ranks.iterrows():
            player_id = row.player_id
            elo_rank_this_year = row.elo_rank
            best_elo_rank = row.bestRank

            atp_ranks_10s.loc[(atp_ranks_10s['player'] == player_id) & (atp_ranks_10s['year'] == year), [
                'elo_rank']] = elo_rank_this_year
            atp_ranks_10s.loc[(atp_ranks_10s['player'] == player_id) & (atp_ranks_10s['year'] == year), [
                'elo_best_rank']] = best_elo_rank

    all_years = [atp_ranks_10s, atp_ranks_00s]
    ranks_all_years = pd.concat(all_years)
    ranks_all_years.to_csv('files/ranks.csv', index=False)


def create_height_track_table():
    """
    Create table of players' heights by date, for easy finding of player's heights in a specific date.
    :return: save the new table in files folder. (the folder contains all necessary files for the system)
    """
    all_dates_and_heights = []
    for year in range(2001, 2022):
        year_matches = pd.read_csv(f'Data/atp_matches/atp_matches_{year}.csv')
        p1_hts = year_matches[['tourney_date', 'winner_id', 'winner_ht']].copy()
        p1_hts.rename(columns={'winner_id': 'player_id', 'winner_ht': 'player_ht'}, inplace=True)
        p2_hts = year_matches[['tourney_date', 'loser_id', 'loser_ht']].copy()
        p2_hts.rename(columns={'loser_id': 'player_id', 'loser_ht': 'player_ht'}, inplace=True)
        all_dates_and_heights += [p1_hts, p2_hts]

    dates_heights_table = pd.concat(all_dates_and_heights)
    dates_heights_table.to_csv('files/heights.csv', index=False)


def create_q1_model():
    """
    Save the fitted model for Q1, for quick response of the system and save run time of it.
    the model can be load successfully on the same python version it's saved.
    The default fitted model comes with the system is for version 3.7.8
    :return: save the fitted model in files/fitted_models folder. (the folder contains  fitted model for all questions)
    """
    train_all_data = pd.read_csv('Data/train/all_years_final_normal_id.csv')
    test_all_data = pd.read_csv('Data/test/test_all_years_final.csv')
    all_data = [train_all_data, test_all_data]
    all_data_table = pd.concat(all_data)
    q1_model = LinearSVC(max_iter=1500, C=10)
    fit_and_save_model(all_data_table, q1_model, 'q1')
    all_data_table.to_csv('files/all_data.csv', index=False)


def create_q2_model():
    """
    Save the fitted model for Q2, for quick response of the system and save run time of it.
    the model can be load successfully on the same python version it's saved.
    The default fitted model comes with the system is for version 3.7.8
    :return: save the fitted model in files/fitted_models folder. (the folder contains  fitted model for all questions)
    """
    train_without_score = pd.read_csv('Data/train/train_without_scores.csv')
    test_without_score = pd.read_csv('Data/test/test_without_scores.csv')
    without_score = [train_without_score, test_without_score]
    without_score_table = pd.concat(without_score)
    q2_model = AdaBoostClassifier(n_estimators=1500)
    fit_and_save_model(without_score_table, q2_model, 'q2')
    without_score_table.to_csv('files/without_score.csv', index=False)


def create_q3_model():
    """
    Save the fitted model for Q3, for quick response of the system and save run time of it.
    the model can be load successfully on the same python version it's saved.
    The default fitted model comes with the system is for version 3.7.8
    :return:  save the fitted model in files/fitted_models folder. (the folder contains  fitted model for all questions)
    """
    train_static_data = pd.read_csv('Data/train/train_static_match_data.csv')
    test_static_data = pd.read_csv('Data/test/test_static_match_data.csv')
    static_data = [train_static_data, test_static_data]
    static_data_table = pd.concat(static_data)
    q3_model = AdaBoostClassifier(n_estimators=350)
    fit_and_save_model(static_data_table, q3_model, 'q3')
    static_data_table.to_csv('files/static_data.csv', index=False)


def fit_and_save_model(data, model, name):
    """
    fit the given model with the given data and save it.
    :param data: DataFrame of all data of the question
    :param model: The chosen model for the question, initialized.
    :param name: name of the question for the saving path
    :return: save the fitted model
    """
    X = data.drop([TARGET], axis=1)
    y = data[TARGET]

    model.fit(X, y)
    joblib.dump(model, f'files/fitted_models/{name}.sav')


def main():
    create_players_table()
    create_ranking_table()
    create_height_track_table()
    create_q1_model()
    create_q2_model()
    create_q3_model()


if __name__ == '__main__':
    main()
