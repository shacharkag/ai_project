import pandas as pd
import matplotlib.pyplot as plt
from create_data_rank_pred import get_first_day_of_q, get_last_day_of_q
from pandas.core.common import SettingWithCopyWarning
import warnings
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import r2_score

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


TARGET = 'p1_won'
Ks = [0, 1, 2, 4, 6, 8, 10, 16, 32, 64, 128]
ALL_QS = [(2005, 1), (2005, 2), (2005, 3), (2005, 4), (2006, 1), (2006, 2), (2006, 3), (2006, 4), (2007, 1), (2007, 2),
          (2007, 3), (2007, 4), (2008, 1), (2008, 2), (2008, 3), (2008, 4), (2009, 1), (2009, 2), (2009, 3), (2009, 4),
          (2010, 1), (2010, 2), (2010, 3), (2010, 4), (2011, 1), (2011, 2), (2011, 3), (2011, 4), (2012, 1), (2012, 2),
          (2012, 3), (2012, 4), (2013, 1), (2013, 2), (2013, 3), (2013, 4), (2014, 1), (2014, 2), (2014, 3), (2014, 4),
          (2015, 1), (2015, 2), (2015, 3), (2015, 4), (2016, 1), (2016, 2), (2016, 3), (2016, 4), (2017, 1), (2017, 2),
          (2017, 3), (2017, 4), (2018, 1), (2018, 2), (2018, 3), (2018, 4), (2019, 1), (2019, 2), (2019, 3), (2019, 4),
          (2020, 1), (2020, 3), (2020, 4), (2021, 1), (2021, 2), (2021, 3)]

fill_missing_rank = {'p1_elo_bestRank': 0.1365559144746962, 'p2_elo_bestRank': 0.14148187741068718,
                     'p1_atp_rank': 0.046167961549591845, 'p2_atp_rank_points': 0.07550748411717913,
                     'p1_elo_rank': 0.26629257885810864, 'p1_atp_rank_points': 0.07481398897918069,
                     'p2_atp_rank': 0.045668815063624385, 'p2_elo_rank': 0.27468119002281255}


def transform_date(date):
    """
    Apply the date transformation we had in all project on the given
    :param date: date to transform
    :return: The numerical value represent the date
    """
    first_date = pd.to_datetime('01/01/2001')
    return (date - first_date).days / 365


def get_first_last_days_of_q_transformed(year, q):
    """
    Get the first dan last date in numerical of the given quarter
    :param year: the year of the quarter
    :param q: the number of the quarter
    :return: first and last days of the q
    """
    first_day_of_q = get_first_day_of_q(year, q)
    last_day_of_q = get_last_day_of_q(year, q)
    first_day_of_q = transform_date(first_day_of_q)
    last_day_of_q = transform_date(last_day_of_q)
    return first_day_of_q, last_day_of_q


def get_last_q_elo_rank(year, q):
    """
    Get the elo ranking table of the quarter before the given quarter
    :param year: the year of the given quarter
    :param q: the number of the given quarter
    :return: elo ranking table of the quarter before the given quarter
    """
    if q == 1:
        year -= 1
        q = 4
    elo_tbl = pd.read_csv(f'Data/Quest4/{year}q{q}.csv')
    return elo_tbl.copy()


def predict_matches_results(data, year, q):
    """
    Predict with the chosen model of Q3 the matches results of every match in the given quarter
    :param data: all matches, processed with normalization, fill missing values exc.
    :param year: the year of the given quarter
    :param q: the number of the given q
    :return: list of 1/-1 results of the model
    """
    first_day_of_q, last_day_of_q = get_first_last_days_of_q_transformed(year, q)

    # Create train from all matches before this quarter, validation is all matches in this quarter
    train = data[data['tourney_date'] < first_day_of_q]
    validation = data[(data['tourney_date'] >= first_day_of_q) & (data['tourney_date'] <= last_day_of_q)].copy()

    # Matches data contains players' rank information in the day of the match. Change all ranks to the last recorded
    # rank before the quarter started
    players_in_q = set(validation['p1_id'].values).union(set(validation['p2_id'].values))
    for player in players_in_q:
        all_player_matches = train[(train['p1_id'] == player) | (train['p2_id'] == player)].copy()
        last_match = all_player_matches.nlargest(1, ['tourney_date'])
        player_side = 1 if len(last_match[last_match['p1_id'] == player]) > 0 else 2
        if len(last_match) > 0:
            last_atp_rank, last_atp_points, last_elo_rank, last_elo_bast_rank = \
                last_match[f'p{player_side}_atp_rank'].iloc[0], last_match[f'p{player_side}_atp_rank_points'].iloc[0], \
                last_match[f'p{player_side}_elo_rank'].iloc[0], last_match[f'p{player_side}_elo_bestRank'].iloc[0]
        else:
            last_atp_rank, last_atp_points, last_elo_rank, last_elo_bast_rank = \
                0.04591838830660812, 0.07516073654817991, 0.2704868844404606, 0.13901889594269168
        validation.loc[validation['p1_id'] == player, 'p1_atp_rank'] = last_atp_rank
        validation.loc[validation['p1_id'] == player, 'p1_atp_rank_points'] = last_atp_points
        validation.loc[validation['p1_id'] == player, 'p1_elo_bestRank'] = last_elo_bast_rank
        validation.loc[validation['p1_id'] == player, 'p1_elo_rank'] = last_elo_rank
        validation.loc[validation['p2_id'] == player, 'p2_atp_rank'] = last_atp_rank
        validation.loc[validation['p2_id'] == player, 'p2_atp_rank_points'] = last_atp_points
        validation.loc[validation['p2_id'] == player, 'p2_elo_bestRank'] = last_elo_bast_rank
        validation.loc[validation['p2_id'] == player, 'p2_elo_rank'] = last_elo_rank

    X_train = train.drop([TARGET], axis=1)
    y_train = train[TARGET]
    X_validation = validation.drop([TARGET], axis=1)

    # Fit and predict
    model = AdaBoostClassifier(n_estimators=350)
    model.fit(X_train, y_train)
    results = model.predict(X_validation)
    return results


def calc_rank_of_q(data, year, q, debug_per_q=False):
    """
    Calculate the elo ranks of the players at the end of the quarter by the predicted results of the matches in the
    quarter. The elo rank is calculated by the start points of the players and increase/decrease in order to matches
    results, whick is also depend on a parameter k (explanation about the formulas in the report).
    In this function, for the given quarter, and predicted results of the matches in it, the elo will be calculate for
    different k values.
    :param data: All matches data
    :param year: The year of the quarter
    :param q: The number of the quarter to predict the rank
    :param debug_per_q: True if plot and print middle debug
    :return: all differences (from the predicted rank ro the real) by each k, a table with predicted rank vs real rank
    """
    # print(f'({year}, {q})')  # for debug
    first_day_of_q, last_day_of_q = get_first_last_days_of_q_transformed(year, q)
    # Find all matches of the quarter to predict their results
    matches_of_q = data[(data['tourney_date'] >= first_day_of_q) & (data['tourney_date'] <= last_day_of_q)].reset_index(
        drop=True)
    # Find the elo points of the players at the start of the quarter
    elo_ranks = get_last_q_elo_rank(year, q)[['player_id', 'points']].copy()
    for k in Ks:
        elo_ranks[f'points{k}'] = elo_ranks['points']
    # Predict the matches results
    matches_results = predict_matches_results(data, year, q)

    # move chronology over the matches in the q, and update the elo points
    for i, match in matches_of_q.iterrows():
        player1 = round(match['p1_id'] * 109369 + 100644)
        player2 = round(match['p2_id'] * 109369 + 100644)
        p1_elo_data = elo_ranks[elo_ranks['player_id'] == player1]
        p2_elo_data = elo_ranks[elo_ranks['player_id'] == player2]
        # If one of the players doesn't have elo points, add him with start of 1500 points
        if len(p1_elo_data) < 1:
            player_data = {'player_id': player1, 'elo_rank': None, 'points': 1500}
            elo_ranks = elo_ranks.append(player_data, ignore_index=True)

        if len(p2_elo_data) < 1:
            player_data = {'player_id': player2, 'elo_rank': None, 'points': 1500}
            elo_ranks = elo_ranks.append(player_data, ignore_index=True)

        # Calculate the updating of the points by the formulas
        p1_elo_points_old = p1_elo_data['points'].iloc[0] if len(p1_elo_data) > 0 else 1500
        p2_elo_points_old = p2_elo_data['points'].iloc[0] if len(p2_elo_data) > 0 else 1500
        p1_won = True if matches_results[i] == 1 else False

        p1_win = 1 / (1 + 10 ** ((p2_elo_points_old - p1_elo_points_old) / 400))
        p2_win = 1 / (1 + 10 ** ((p1_elo_points_old - p2_elo_points_old) / 400))

        p1_outcome = 1 if p1_won else 0
        p2_outcome = 0 if p1_won else 1

        for k in Ks:
            p1_elo_points_new = p1_elo_points_old + k * (p1_outcome - p1_win)
            p2_elo_points_new = p2_elo_points_old + k * (p2_outcome - p2_win)

            # Update the new points at the table for using them again when the players will play
            elo_ranks.loc[elo_ranks['player_id'] == player1, f'points{k}'] = p1_elo_points_new
            elo_ranks.loc[elo_ranks['player_id'] == player2, f'points{k}'] = p2_elo_points_new

    # Get the real elo rank at the end of the q. This tabe is sorted from the first rank to the last
    final_q_elo_rank = pd.read_csv(f'Data/Quest4/{year}q{q}.csv')
    real_ranks = dict()
    for i, player_elo_data in final_q_elo_rank.iterrows():
        real_ranks[player_elo_data.loc['player_id']] = i + 1

    hit_hist = {k: [] for k in Ks}
    player_predict_vs_real_rank = {k: pd.DataFrame() for k in Ks}

    for k in Ks:
        # Sort in decreasing order by the points we calculated, the player with the highest points will got the 1st rank
        sorted_elo = elo_ranks.sort_values(by=[f'points{k}'], ascending=False)
        sorted_elo = sorted_elo.reset_index(drop=True)
        calculated_ranks = dict()
        for i, player_elo_data in sorted_elo.iterrows():
            calculated_ranks[player_elo_data.loc['player_id']] = i + 1

        # Compare the predicted rank with the real rank and calculate the difference between them
        for player in calculated_ranks.keys():
            if player in real_ranks.keys():
                player_predict_vs_real_rank[k] = player_predict_vs_real_rank[k].append(
                    {'player_id': player, 'predicted': calculated_ranks[player], 'real': real_ranks[player]},
                    ignore_index=True)
                if calculated_ranks[player] == real_ranks[player]:
                    if debug_per_q:
                        print(f'WOW hit player {player}!')
                    hit_hist[k].append(0)
                else:
                    difference = real_ranks[player] - calculated_ranks[player]
                    if debug_per_q:
                        print(f'On player {player} miss with {difference}')
                    hit_hist[k].append(difference)

        if debug_per_q:
            # Plot histograms of the differences
            bins = max(hit_hist[k]) + abs(min(hit_hist[k])) + 1
            plt.hist(hit_hist[k], bins=bins)
            plt.title(f'year: {year}, q: {q}, k={k}')
            plt.show()

        # Print the r2 score of the q rank calculation
        print(f"R2 for year: {year}, q: {q}, k={k} is "
              f"{r2_score(player_predict_vs_real_rank[k]['real'], player_predict_vs_real_rank[k]['predicted'])}")

        sorted_elo.to_csv(f'files/Question_4_final_results/final_rank{year}q{q}k{k}.csv')
    return hit_hist, player_predict_vs_real_rank


def main():
    # Load all matches after all processing of 3rd question
    all_matches_data = pd.read_csv('Data/Quest4/all_matches_static_data.csv')

    # Initiate dict to save results per k
    all_players_predict_vs_real_rank = {k: [] for k in Ks}
    all_differences = {k: [] for k in Ks}

    for q in ALL_QS:
        differences, predicts_vs_reals = calc_rank_of_q(all_matches_data, q[0], q[1], debug_per_q=True)
        for k in Ks:
            all_differences[k] += differences[k]
            all_players_predict_vs_real_rank[k].append(predicts_vs_reals[k])

    # gather all results of each k, and plot histogram of all differences and print the r2 score
    for k in Ks:
        bins = max(all_differences[k]) + abs(min(all_differences[k])) + 1
        plt.hist(all_differences[k], bins=bins)
        plt.title(f'All quarterlies differences of ranks k={k}')
        plt.show()
        df_predict_vs_real_rank = pd.concat(all_players_predict_vs_real_rank[k])
        print('*************************')
        print(f"R2 for k={k} is {r2_score(df_predict_vs_real_rank['real'], df_predict_vs_real_rank['predicted'])}")
        print('*************************')


if __name__ == '__main__':
    main()