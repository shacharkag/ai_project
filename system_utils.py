import pandas as pd


class Player:
    def __init__(self, players_data: pd.DataFrame, player_name: str, data_to_fill: dict, ranks: pd.DataFrame,
                 heights: pd.DataFrame, p):
        match_date = pd.to_datetime(data_to_fill['tourney_date'][0])
        # Find player ID, hand by name
        data_to_fill[f'p{p}_id'] = [int(players_data[players_data['full_name'] == player_name]['player_id'])]
        data_to_fill[f'p{p}_hand'] = [players_data[players_data['full_name'] == player_name]['player_hand'].iloc[0]]
        # Calculate player age at the match day
        date_of_birth = pd.to_datetime(players_data[players_data['full_name'] == player_name]['date_of_birth'].iloc[0], format='%Y%m%d')
        age_in_days = (match_date - date_of_birth)
        data_to_fill[f'p{p}_age'] = [age_in_days.days / 365]

        # Fill elo rank data
        player_ranks = ranks[ranks['player'] == data_to_fill[f'p{p}_id'][0]]
        if player_ranks[player_ranks['year'] == match_date.year]['elo_rank'].count() > 0:
            data_to_fill[f'p{p}_elo_rank'] = [int(player_ranks[player_ranks['year'] == match_date.year]['elo_rank'].iloc[0])]
            data_to_fill[f'p{p}_elo_bestRank'] = [int(player_ranks[player_ranks['year'] == match_date.year]['elo_best_rank'].iloc[0])]
        else:
            data_to_fill[f'p{p}_elo_rank'] = data_to_fill[f'p{p}_elo_bestRank'] = [250]

        # Fill atp rank data (the last rank reported on the match day and earlier)
        player_ranks['ranking_date'] = pd.to_datetime(player_ranks['ranking_date'])
        only_earlier_dates = player_ranks[player_ranks['ranking_date'] <= match_date]
        possible_dates = only_earlier_dates['ranking_date'].values
        if len(possible_dates) > 0:
            closest_date = nearest(possible_dates, match_date)
            data_to_fill[f'p{p}_atp_rank'] = [player_ranks[player_ranks['ranking_date'] == closest_date]['atp_rank'].iloc[0]]
            data_to_fill[f'p{p}_atp_rank_points'] = [player_ranks[player_ranks['ranking_date'] == closest_date]['atp_points'].iloc[0]]
        else:
            data_to_fill[f'p{p}_atp_rank'] = data_to_fill[f'p{p}_atp_rank_points'] = [None]

        # Find the relevant player's height for the match (the last height reported on the match day and earlier)
        player_hts = heights[heights['player_id'] == data_to_fill[f'p{p}_id'][0]]
        player_hts['tourney_date'] = pd.to_datetime(player_hts['tourney_date'])
        only_earlier_dates = player_hts[player_hts['tourney_date'] <= match_date]
        possible_dates = only_earlier_dates['tourney_date'].values
        if len(possible_dates) > 0:
            closest_date = nearest(possible_dates, match_date)
            data_to_fill[f'p{p}_ht'] = [player_hts[player_hts['tourney_date'] == closest_date]['player_ht'].iloc[0]]
        else:
            data_to_fill[f'p{p}_ht'] = [None]


def nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))
