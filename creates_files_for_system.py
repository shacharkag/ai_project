import pandas as pd
import numpy as np


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

    atp_ranks_00s.rename(columns={'rank': 'atp_rank', 'points': 'atp_points'}, inplace=True)
    atp_ranks_10s.rename(columns={'rank': 'atp_rank', 'points': 'atp_points'}, inplace=True)

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


def main():
    create_players_table()
    create_ranking_table()
    create_height_track_table()


if __name__ == '__main__':
    main()
