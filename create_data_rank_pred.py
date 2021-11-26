import pandas as pd
import datetime
from built_data import get_player_id_by_full_name
from system_utils import nearest


FIRST_MONTH_OF_Q = {'1': 1, '2': 4, '3': 7, '4': 10}
LAST_MONTH_OF_Q = {'1': 3, '2': 6, '3': 9, '4': 12}
LAST_DAY_OF_Q = {'1': 31, '2': 30, '3': 30, '4': 31}


def get_last_day_of_q(year, q):
    # return pd.datetime of the last day of a quarterly
    return pd.to_datetime(datetime.date(year, LAST_MONTH_OF_Q[str(q)], LAST_DAY_OF_Q[str(q)]))


def get_first_day_of_q(year, q):
    # return pd.datetime of the first day of a quarterly
    return pd.to_datetime(datetime.date(year, FIRST_MONTH_OF_Q[str(q)], 1))


def get_first_day_of_year(year):
    # return pd.datetime of the first day of a year
    return pd.to_datetime(datetime.date(year, 1, 1))


def get_last_day_of_year(year):
    # return pd.datetime of the last day of a year
    return pd.to_datetime(datetime.date(year, 12, 31))


def add_id_to_elo_tbl():
    """
    for every elo table, (have one for every quarterly), add column of the player ID, by match the player name in the
    elo table to the relevant players table.
    Moreover find the max elo rank, to use it later for filling ranks for players that don't have reported rank.
    :return: Save the new tables to csv files
    """
    players = pd.read_csv('Data/relevant_players.csv')
    players['first_name'] = players['first_name'].str.lower()
    players['last_name'] = players['last_name'].str.lower()
    players['full_name'] = players['first_name'] + ' ' + players['last_name']

    max_elo = None

    for year in range(2000, 2022):
        for q in range(1, 5):
            if (year == 2020 and q == 2) or (year == 2021 and q == 4):
                # second 2020 quarterly is missing because of covid-19,
                # forth 2021 quarterly is yet to be done
                continue
            elo_tbl = pd.read_csv(f'Data/elo_ranking/Quest4/Rankings{year}q{q}.csv', encoding='ISO-8859-1')
            elo_tbl = elo_tbl.rename(columns={'rank': 'elo_rank'})
            elo_tbl['name'] = elo_tbl['name'].str.lower()
            # Add ID for each row in elo table
            elo_tbl['player_id'] = elo_tbl['name'].apply(lambda row: get_player_id_by_full_name(row, players, year))
            # Drop irrelevant features for the regression
            elo_tbl.drop(['country_name', 'country_id'], axis='columns', inplace=True)

            # Find the max rank reported for later
            rank_count = max(elo_tbl['elo_rank'].values)
            if max_elo is None or rank_count > max_elo:
                max_elo = rank_count

            print(f' for {year}: num of missing ids: {elo_tbl.player_id.isna().sum()}')

            elo_tbl.to_csv(f'Data/Quest4/{year}q{q}.csv', index=False)
    print(f'max elo is: {max_elo}')


def find_all_players_ranks():
    """
    For players who played in a q, but don't have reported elo rank in the end of the q, generate a elo rank by their
    atp rank. Find all players have atp rank and don't have elo rank, and order them by the atp rank, and give them by
    this order increasing rank, start in 250.
    :return:
    """
    for year in range(2000, 2022):
        for q in range(1, 5):
            if (year == 2020 and q == 2) or (year == 2021 and q == 4):
                continue
            elo_tbl = pd.read_csv(f'Data/Quest4/{year}q{q}.csv')
            elo_tbl.drop(['name'], axis='columns', inplace=True)
            # All players IDs who has reported elo rank in the end of this q
            players_have_elo = set(elo_tbl['player_id'].values)
            matches_in_this_q = get_matches_by_year_and_q(year, q)
            # All players IDs who played in this q
            players_in_matches = set(matches_in_this_q['winner_id'].values).union(matches_in_this_q['loser_id'].values)
            # Subtracts groups, find all players without reported elo rank
            players_without_elo = players_in_matches - players_have_elo
            # print(f'for {year}q{q}: {players_without_elo=}')  # For debugging

            # Find for the players with missing elo rank, their last recorded atp rank
            players_without_elo_with_atp = get_last_atp_rank_for_missing_elo(year, q, players_without_elo)
            # Generate increasing elo for the best atp rank to the lowest, starts from elo 250
            for i, rank in enumerate(sorted(players_without_elo_with_atp.keys())):
                for player in players_without_elo_with_atp[rank]:
                    elo_tbl = elo_tbl.append(
                        {'elo_rank': 250 + i, 'player_id': player, 'points': None, 'bestRank': 250 + i,
                         'rankDiff': None, 'pointsDiff': None, 'bestPoints': None}, ignore_index=True)

            elo_tbl.to_csv(f'Data/elo_ranking/Quest4/ranks_fill_missing_players{year}q{q}.csv', index=False)


def get_last_atp_rank_for_missing_elo(year, q, set_of_players):
    """
    For every player without elo rank, find his last atp rank (the last atp rank that was recorded at the q or before)
    :param year: The year of the q
    :param q: number of quarter
    :param set_of_players: IDs of players without elo rank recorded
    :return: dict in format: <rank>: [<playerID>]
    """
    # Init dict to return
    ranks_of_players = dict()
    ranks_list = pd.read_csv('files/ranks.csv')
    last_day_of_q = get_last_day_of_q(year, q)
    for player in set_of_players:
        # Gat all atp rank recordings of a player
        player_ranks = ranks_list[ranks_list['player'] == player]
        player_ranks['ranking_date'] = pd.to_datetime(player_ranks['ranking_date'])
        # Filter only recording of this q or earlier
        only_earlier_dates = player_ranks[player_ranks['ranking_date'] <= last_day_of_q]
        possible_dates = only_earlier_dates['ranking_date'].values
        if len(possible_dates) > 0:
            # Find the closest record of rank to the end of the q
            closest_date = nearest(possible_dates, last_day_of_q)
            p_rank = player_ranks[player_ranks['ranking_date'] == closest_date]['atp_rank'].iloc[0]
            if p_rank in ranks_of_players.keys():
                ranks_of_players[p_rank].append(player)
            else:
                ranks_of_players[p_rank] = [player]

    return ranks_of_players


def apply_func_all_years(func):
    """
    general function to apply aothe function over all Qs fromm 2000.
    :param func: The function to apply
    """
    for year in range(2000, 2022):
        for q in range(1, 5):
            if (year == 2020 and q == 2) or (year == 2021 and q == 4):
                continue
            func(year, q)


def fill_players_statistics(year, q):
    """
    Calculate players statistic of the q.
    :param year: year of the q
    :param q: number of the quarter
    :return: save the statistics table to csv file
    """
    elo_tbl = pd.read_csv(f'Data/elo_ranking/Quest4/ranks_fill_missing_players{year}q{q}.csv')
    matches_this_q = get_matches_by_year_and_q(year, q)
    players_had_match_this_q = set(matches_this_q['winner_id'].values).union(matches_this_q['loser_id'].values)
    all_players_in_q = set(elo_tbl['player_id'])
    players_statistic = pd.DataFrame()
    for player in all_players_in_q:
        player_stat = {'player_id': player, 'year': year % 100, 'q': q / 4}
        player_stat['age'] = get_player_age(player, year, q)
        player_stat['height'] = get_player_height(player, year, q)
        if player in players_had_match_this_q:
            player_stat['num_of_wins'] = matches_this_q[matches_this_q['winner_id'] == player].shape[0]
            player_stat['num_of_participent'] = player_stat['num_of_wins'] + matches_this_q[matches_this_q['loser_id'] == player].shape[0]
            player_stat['num_of_wins_on_grass'] = matches_this_q[(matches_this_q['winner_id'] == player) &
                                                  (matches_this_q['surface'] == 'Grass')].shape[0]
            player_stat['num_of_wins_on_clay'] = matches_this_q[
                (matches_this_q['winner_id'] == player) & (matches_this_q['surface'] == 'Clay')].shape[0]
            player_stat['num_of_wins_on_carpet'] = matches_this_q[
                (matches_this_q['winner_id'] == player) & (matches_this_q['surface'] == 'Carpet')].shape[0]
            player_stat['num_of_wins_on_hard'] = matches_this_q[
                (matches_this_q['winner_id'] == player) & (matches_this_q['surface'] == 'Hard')].shape[0]
            player_stat['num_of_participent_on_grass'] = player_stat['num_of_wins_on_grass'] + matches_this_q[
                (matches_this_q['loser_id'] == player) & (matches_this_q['surface'] == 'Grass')].shape[0]
            player_stat['num_of_participent_on_clay'] = player_stat['num_of_wins_on_clay'] + matches_this_q[
                (matches_this_q['loser_id'] == player) & (matches_this_q['surface'] == 'Clay')].shape[0]
            player_stat['num_of_participent_on_carpet'] = player_stat['num_of_wins_on_carpet'] + matches_this_q[
                (matches_this_q['loser_id'] == player) & (matches_this_q['surface'] == 'Carpet')].shape[0]
            player_stat['num_of_participent_on_hard'] = player_stat['num_of_wins_on_hard'] + matches_this_q[
                (matches_this_q['loser_id'] == player) & (matches_this_q['surface'] == 'Hard')].shape[0]
            player_stat['wins_players_with_lower_atp_rank'] = matches_this_q[(matches_this_q['winner_id'] == player)
                                        & (matches_this_q['winner_rank'] > matches_this_q['loser_rank'])].shape[0]
            player_stat['loss_players_with_lower_atp_rank'] = matches_this_q[(matches_this_q['loser_id'] == player)
                                        & (matches_this_q['winner_rank'] < matches_this_q['loser_rank'])].shape[0]

            performance_features = ['ace', 'df', 'svpt', '1stIn', '1stWon', '2ndWon', 'SvGms', 'bpSaved', 'bpFaced']
            for feature in performance_features:
                w_value = list(matches_this_q[(matches_this_q['winner_id'] == player)][f'w_{feature}'].values)
                l_value = list(matches_this_q[(matches_this_q['loser_id'] == player)][f'l_{feature}'].values)
                values = w_value + l_value
                player_stat[feature] = sum(values) / len(values)

        print(player_stat)
        players_statistic = players_statistic.append(player_stat, ignore_index=True)
    result = pd.merge(elo_tbl, players_statistic, how="left", on='player_id', indicator=True)
    result.to_csv(f'Data/Quest4/players_statistics_{year}q{q}.csv', index=False)


def get_matches_by_year_and_q(year, q):
    """
    Get all matches occurred at this q
    :param year: the year of the q
    :param q: the number of the quarter
    :return: DataFrame of all matches data of this q
    """
    first_day_of_q = get_first_day_of_q(year, q)
    last_day_of_q = get_last_day_of_q(year, q)
    matches = [pd.read_csv(f'Data/atp_matches/atp_matches_{year}.csv')]
    if q == 1 and year != 2000:
        matches.append(pd.read_csv(f'Data/atp_matches/atp_matches_{year-1}.csv'))
    if q == 4 and year != 2021:
        matches.append(pd.read_csv(f'Data/atp_matches/atp_matches_{year + 1}.csv'))
    all_matches = pd.concat(matches)

    all_matches['tourney_date'] = pd.to_datetime(all_matches['tourney_date'], format='%Y%m%d')
    matches_in_q = all_matches[(all_matches['tourney_date'] >= first_day_of_q) &
                               (all_matches['tourney_date'] <= last_day_of_q)]
    return matches_in_q


def get_player_age(player, year, q):
    """
    Calculate player age at the last day of the q
    :param player: player ID
    :param year: year of the q
    :param q: number of the quarter
    :return: the player age
    """
    players = pd.read_csv('Data/relevant_players.csv')
    last_day_of_q = get_last_day_of_q(year, q)
    date_of_birth = pd.to_datetime(players[players['player_id'] == player]['date_of_birth'].iloc[0], format='%Y%m%d')
    age_in_days = (last_day_of_q - date_of_birth)
    return age_in_days.days / 365


def get_player_height(player, year, q):
    """
    Find the height of a player at the querter, by find the last player's height record before the end of the q.
    :param player: player ID
    :param year: year of the q
    :param q: number of the quarter
    :return: player height found
    """
    heights_list = pd.read_csv('files/heights.csv')
    last_day_of_q = get_last_day_of_q(year, q)
    player_hts = heights_list[heights_list['player_id'] == player]
    player_hts['tourney_date'] = pd.to_datetime(player_hts['tourney_date'])
    only_earlier_dates = player_hts[player_hts['tourney_date'] <= last_day_of_q]
    possible_dates = only_earlier_dates['tourney_date'].values
    if len(possible_dates) > 0:
        closest_date = nearest(possible_dates, last_day_of_q)
        return player_hts[player_hts['tourney_date'] == closest_date]['player_ht'].iloc[0]
    else:
        return None


def transform_date_and_fill_missing_data(year, q):
    """
    Process the quarter's DataFrame.
    Transform the 'bestRankDate' values from date to number by calculate the number of days passed from 01/01/2000 and
    divide by 365.
    fill missing values with 0.
    Drop irrelevant features.
    :param year: the year of the q
    :param q: number of the quarter
    :return: save the processed table to csv file
    """
    player_data_this_year = pd.read_csv(f'Data/Quest4/players_statistics_{year}q{q}.csv')
    first_date = pd.to_datetime('01/01/2000')
    player_data_this_year['bestRankDate'].fillna(value=first_date, inplace=True)
    player_data_this_year['bestRankDate'] = pd.to_datetime(player_data_this_year['bestRankDate'])
    player_data_this_year['bestRankDate'] = player_data_this_year['bestRankDate'].apply(
        lambda x: (x - first_date).days / 365)
    player_data_this_year.fillna(0, inplace=True)
    player_data_this_year.drop(['points', 'rankDiff', 'pointsDiff', 'bestPoints'], axis='columns', inplace=True)

    player_data_this_year.to_csv(f'Data/Quest4/players_data_filling_data_{year}q{q}.csv', index=False)


def built_big_table():
    """
    Concat all tables of quartiles together. apply min-max normalization for getting one scale for all features.
    :return: save the big table to csv file
    """
    all_years = []
    for year in range(2000, 2022):
        for q in range(1, 5):
            if (year == 2020 and q == 2) or (year == 2021 and q == 4):
                continue
            all_years.append(pd.read_csv(f'Data/Quest4/players_data_filling_data_{year}q{q}.csv'))
    all_years_df = pd.concat(all_years)
    all_years_df.to_csv(f'Data/Quest4/all_years_init.csv', index=False)

    mins = dict()
    maxs = dict()

    feature_to_minmax = {'bestRankDate', 'player_id', 'age', 'height', 'year', '1stIn', '1stWon', '2ndWon',
                         'bpFaced', 'bpSaved', 'df',	'loss_players_with_lower_atp_rank',  'SvGms', 'ace',
                         'num_of_participent', 'num_of_participent_on_carpet', 'num_of_participent_on_clay',
                         'num_of_participent_on_grass', 'num_of_participent_on_hard', 'num_of_wins',
                         'num_of_wins_on_carpet', 'num_of_wins_on_clay', 'num_of_wins_on_grass',
                         'num_of_wins_on_hard', 'svpt', 'wins_players_with_lower_atp_rank'}

    for feature in feature_to_minmax:
        min_val = mins[feature] = all_years_df[feature].min()
        max_val = maxs[feature] = all_years_df[feature].max()
        all_years_df[feature] = (all_years_df[feature] - min_val) / (max_val - min_val)

    all_years_df['elo_rank'] = (all_years_df['elo_rank'] - 1) / (249)
    all_years_df['bestRank'] = (all_years_df['bestRank'] - 1) / (249)

    print(f'mins dict:\n{mins} \n\n maxs dict:\n{maxs}')
    all_years_df.to_csv(f'Data/Quest4/all_years_after_min_max_elo.csv', index=False)


def create_list_of_all_players_of_year():
    """
    Print all players plays at every year. manually save the dictionary at file players_plays_per_year.py
    """
    players_plays_per_year = dict()
    for year in range(2000, 2022):
        first_day_of_year = get_first_day_of_year(year)
        last_day_of_year = get_last_day_of_year(year)
        matches = [pd.read_csv(f'Data/atp_matches/atp_matches_{year}.csv')]
        if year != 2000:
            matches.append(pd.read_csv(f'Data/atp_matches/atp_matches_{year - 1}.csv'))
        if year != 2021:
            matches.append(pd.read_csv(f'Data/atp_matches/atp_matches_{year + 1}.csv'))
        all_matches = pd.concat(matches)

        all_matches['tourney_date'] = pd.to_datetime(all_matches['tourney_date'], format='%Y%m%d')
        matches_in_year = all_matches[(all_matches['tourney_date'] >= first_day_of_year) &
                                   (all_matches['tourney_date'] <= last_day_of_year)]
        players_plays_per_year[str(year)] = set(matches_in_year['winner_id'].values).union(set(matches_in_year['loser_id'].values))

    print(players_plays_per_year)


def create_tbl_of_all_matches_static_data():
    """
    Create table of all static data of matches from 2001. The source tables, come from the processed train and test
    tables for the 3rd question of this project.
    :return: save the new table as csv file
    """
    matches = [pd.read_csv('Data/train/train_static_match_data.csv'),
               pd.read_csv('Data/test/test_static_match_data.csv')]

    all_matches = pd.concat(matches)
    all_matches.sort_values(by=['tourney_date'], inplace=True)

    all_matches.to_csv('Data/Quest4/all_matches_static_data.csv', index=False)


def main():
    add_id_to_elo_tbl()
    find_all_players_ranks()
    apply_func_all_years(fill_players_statistics)
    apply_func_all_years(transform_date_and_fill_missing_data)
    built_big_table()
    create_list_of_all_players_of_year()
    create_tbl_of_all_matches_static_data()


if __name__ == '__main__':
    main()
