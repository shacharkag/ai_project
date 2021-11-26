import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import os


PROBLEMATIC_NAMES =\
    {'viloca j.a.': 102167, 'damm m.': 210013, 'lee h.t.': 102703, 'bahrouzyan o.': 103914, 'nadal-parera r.': 104745,
     'andersen j.f.': 102107, 'vassallo-arguello m.': 103506, 'ivanov-smolensky k.': 103715, 'bogomolov jr.a.': 104166,
     'elgin m.': 103862, 'silva f.': 106393, 'gambill j. m.': 102998, 'dell\'acqua m.': 103435, 'menezes j.': 110778,
     'garcia-lopez g.': 104198, 'robredo r.': 103990, 'ramirez-hidalgo r.': 103105, 'bogomolov a.': 104166,
     'o\'brien a.': 101711, 'goodall j.': 104623, 'al ghareeb m.': 103600, 'matos-gil i.': 104099, 'galan d.': 123755,
     'navarro-pastor i.': 103868, 'gruber k.': 103460, 'harper-griffith l.': 103834, 'herm-zahlava j.': 103564,
     'garcia-sintes j.': 103365, 'al-alawi s.k.': 102956, 'bogomolov jr. a.': 104166, 'kunitcin i.': 103857,
     'hantschek m.': 103082, 'ratiwatana s.': 103920, 'roger-vasselin e.': 104273, 'gallardo m.': 103746,
     'gimeno d.': 104593, 'gimeno-traver d.': 104593, 'wang y. jr': 103817, 'querry s.': 105023, 'mecir m.': 105080,
     'haider-mauer a.': 104890, 'sultan-khalfan a.': 102956, 'king-turner d.': 104368, 'van der dium a.': 105014,
     'rascon t.': 101869, 'granollers-pujol m.': 104719, 'salva b.': 104831, 'luque d.': 104317, 'vicente m.': 102950,
     'dev varman s.': 104500, 'fish a.': 103888, 'stepanek m.': 103285, 'fornell m.': 103924, 'guccione a.': 104589,
     'chekov p.': 105189, 'trujillo g.': 103448, 'statham r.': 104907, 'jones g.d.': 104711, 'podlipnik h.': 105070,
     'al-ghareeb m.': 103600, 'nader m.': 108996, 'lopez-jaen m.a.': 104091, 'o\'neal j.': 103076, 'haji a.': 108993,
     'de heart r.': 104281, 'ruevski p.': 104234, 'riba-madrid p.': 105137, 'munoz-de la nava d.': 103926,
     'del bonis f.': 105643, 'haider-maurer a.': 104890, 'ramos-vinolas a.': 105077, 'dolgopolov o.': 105238,
     'granollers pujol g.': 105283, 'gutierrez-ferrol s.': 105300, 'carreno-busta p.': 105807, 'bautista r.': 105138,
     'dutra da silva r.': 104297, 'granollers-pujol g.': 105283, 'cervantes i.': 105438, 'van d. merwe i.': 104292,
     'brugues-davi a.': 104516, 'gomez-herrera c.': 105529, 'menendez-maceiras a.': 104629, 'deen heshaam a.': 106277,
     'ali mutawa j.m.': 106325, 'zayed m. s.': 110476, 'silva f.f.': 106393, 'reyes-varela m.a.': 104960,
     'estrella burgos v.': 103607, 'nedovyesov o.': 104873, 'artunedo martinavarro a.': 106238, 'galan d.e.': 123755,
     'prashanth v.': 104822, 'ortega-olmedo r.': 105751, 'zayid m. s.': 122570, 'samper-montana j.': 105515,
     'lopez-perez e.': 105782, 'del potro j. m.': 105223, 'o\'connell c.': 106331, 'alcaraz c.': 207989,
     'alawadhi o.': 200525, 'hernandez-fernandez j.': 105497, 'auger-aliassime f.': 200000, 'o connell c.': 106331,
     'varillas j.p.': 122669, 'varillas j. p.': 122669, 'hernandez-fernandez j': 105497, 'meligeni rodrigues f': 200335,
     'barrios m.': 144642}

DUPLICATE_NAMES_OF_2001 = {'martin a.': 103252, 'ruud c.': 102104, 'alvarez e.': 102135, 'petrovic d.': 103162,
                           'wang y.': 103817}
DUPLICATE_NAMES_OF_2002 = {'martin a.': 103252, 'marin l.': 102192, 'albert m.': 102796}
DUPLICATE_NAMES_OF_2003 = {'martin a.': 103252, 'schuettler p.': 102783, 'youzhny a.': 104022, 'prpic a.': 104004,
                           'ascione a.': 103693, 'andersen j.': 102107}
DUPLICATE_NAMES_OF_2004 = {'martin a.': 103252, 'kucera v.': 102344, 'ancic i.': 104339, 'verdasco m.': 104269,
                           'wang y.t.': 104499, 'benneteau a.': 103898, 'wang y.': 103817}
DUPLICATE_NAMES_OF_2005 = {'wang y.t.': 104499, 'martin a.': 103252, 'jones g.d.': 104711, 'march o.': 103594,
                           'kuznetsov a.': 104864}
DUPLICATE_NAMES_OF_2006 = {'martin a.': 103252, 'wang y.t.': 104499, 'kuznetsov a.': 104864, 'prpic a.': 104004,
                           'wang y.': 104499, 'kutac r.': 104519, 'wang y.jr.': 103817, 'schuttler p.': 102783}
DUPLICATE_NAMES_OF_2007 = {'martin a.': 103252, 'kuznetsov a.': 104864, 'wang y.t.': 104499, 'youzhny a.': 104022}
DUPLICATE_NAMES_OF_2008 = {'jones g.': 105286, 'martin a.': 103252}
DUPLICATE_NAMES_OF_2009 = {'martin a.': 103252, 'kuznetsov a.': 105723, 'silva d.': 105180, 'luncanu p.a.': 105331}
DUPLICATE_NAMES_OF_2010 = {'maamoun k.': 103472, 'zhang z.': 105585, 'kuznetsov a.': 105723, 'martin a.': 103252,
                           'martin an.': 105413}
DUPLICATE_NAMES_OF_2011 = {'kuznetsov an.': 105723, 'kuznetsov a.': 105723, 'kuznetsov al.': 104864, 'zhang z.': 105585,
                           'jones g.': 105286, 'sanchez m.': 105688}
DUPLICATE_NAMES_OF_2012 = {'jones g.': 105286, 'zhang z.': 105585, 'kuznetsov an.': 105723, 'kuznetsov a.': 104864,
                           'kuznetsov al.': 104864}
DUPLICATE_NAMES_OF_2013 = {'jones g.': 105286, 'kuznetsov an.': 105723, 'kuznetsov al.': 104864, 'chung h.': 111202,
                           'martin a.': 105413, 'zhang z.': 105585}
DUPLICATE_NAMES_OF_2014 = {'kuznetsov an.': 105723, 'kuznetsov al.': 104864, 'zhang z.': 105585, 'wang c.': 105934,
                           'martin a.': 105413}
DUPLICATE_NAMES_OF_2015 = {'chung h.': 111202, 'kuznetsov an.': 105723, 'kuznetsov al.': 104864, 'zhang ze': 105585,
                           'zhang zh.': 111190, 'zhang z.': 105585}
DUPLICATE_NAMES_OF_2016 = {'chung h.': 111202,  'kuznetsov an.': 105723, 'kuznetsov al.': 104864, 'martin a.': 105413,
                           'zhang ze': 105585, 'ruud c.': 134770}
DUPLICATE_NAMES_OF_2017 = {'chung h.': 111202, 'kuznetsov an.': 105723, 'kuznetsov al.': 104864, 'ruud c.': 134770,
                           'martin a.': 105413, 'zhang zh.': 111190, 'sanchez m.': 105688, 'zhang ze': 105585}
DUPLICATE_NAMES_OF_2018 = {'chung h.': 111202, 'ruud c.': 134770, 'martin a.': 105413, 'martinez p.': 124079,
                           'zhang ze': 105585, 'zhang zh.': 111190}
DUPLICATE_NAMES_OF_2019 = {'chung h.': 111202, 'ruud c.': 134770, 'zhang zh.': 111190, 'zhang ze.': 105585,
                           'martin a.': 105413, 'martinez p.': 124079, 'monteiro j.': 106329}
DUPLICATE_NAMES_OF_2020 = {'martin a.': 105413, 'martinez p.': 124079, 'ruud c.': 134770, 'kuznetsov an.': 105723,
                           'kuznetsov al.': 104864, 'petrovic d.': 105905}
DUPLICATE_NAMES_OF_2021 = {'martinez p.': 124079, 'ruud c.': 134770, 'martin a.': 105413, 'zhang zh.': 111190,
                           'zhang ze.': 105585, 'petrovic d.': 105905, 'kuznetsov an.': 105723,}

DICT_DUPLICATE_BY_YEAR = {'2001': DUPLICATE_NAMES_OF_2001, '2002': DUPLICATE_NAMES_OF_2002,
                          '2003': DUPLICATE_NAMES_OF_2003, '2004': DUPLICATE_NAMES_OF_2004,
                          '2005': DUPLICATE_NAMES_OF_2005, '2006': DUPLICATE_NAMES_OF_2006,
                          '2007': DUPLICATE_NAMES_OF_2007, '2008': DUPLICATE_NAMES_OF_2008,
                          '2009': DUPLICATE_NAMES_OF_2009, '2010': DUPLICATE_NAMES_OF_2010,
                          '2011': DUPLICATE_NAMES_OF_2011, '2012': DUPLICATE_NAMES_OF_2012,
                          '2013': DUPLICATE_NAMES_OF_2013, '2014': DUPLICATE_NAMES_OF_2014,
                          '2015': DUPLICATE_NAMES_OF_2015, '2016': DUPLICATE_NAMES_OF_2016,
                          '2017': DUPLICATE_NAMES_OF_2017, '2018': DUPLICATE_NAMES_OF_2018,
                          '2019': DUPLICATE_NAMES_OF_2019, '2020': DUPLICATE_NAMES_OF_2020,
                          '2021': DUPLICATE_NAMES_OF_2021}

ELO_PROBLAMETIC_NAMES = \
    {'martin damm': 210013, 'stan wawrinka': 104527, 'frances tiafoe': 126207, 'boris becker': 101414,
     'petr korda': 101434, 'christophe van garsse': 102419, 'brett steven': 101601, 'sandon stolle': 101776,
     'david nainkin': 101805, 'danny sapsford': 101591, 'chris wilkinson': 101681, 'geoff grant': 101688,
     'rodolphe gilbert': 101549, 'carlos costa': 119129, 'joseph lizardo': 101563, 'donald johnson': 101515,
     'borut urh': 102436, 'hideki kaneko': 102348, 'marco meneschincheri': 102050}

RENAME_DICT = \
    {'winner_id': 'p1_id', 'winner_hand': 'p1_hand', 'winner_ht': 'p1_ht', 'winner_age': 'p1_age', 'loser_id': 'p2_id',
     'loser_hand': 'p2_hand', 'loser_ht': 'p2_ht', 'loser_age': 'p2_age', 'w_ace': 'p1_ace', 'w_df': 'p1_df',
     'w_svpt': 'p1_svpt', 'w_1stIn': 'p1_1stIn', 'w_1stWon': 'p1_1stWon', 'w_2ndWon': 'p1_2ndWon',
     'w_SvGms': 'p1_SvGms', 'w_bpSaved': 'p1_bpSaved', 'w_bpFaced': 'p1_bpFaced', 'l_ace': 'p2_ace', 'l_df': 'p2_df',
     'l_svpt': 'p2_svpt', 'l_1stIn': 'p2_1stIn', 'l_1stWon': 'p2_1stWon', 'l_2ndWon': 'p2_2ndWon',
     'l_SvGms': 'p2_SvGms', 'l_bpSaved': 'p2_bpSaved', 'l_bpFaced': 'p2_bpFaced', 'winner_rank': 'p1_atp_rank',
     'winner_rank_points': 'p1_atp_rank_points', 'loser_rank': 'p2_atp_rank', 'loser_rank_points': 'p2_atp_rank_points',
     'elo_rank_winner': 'p1_elo_rank', 'elo_points_winner': 'p1_elo_points', 'elo_bestRank_winner': 'p1_elo_bestRank',
     'elo_bestPoints_winner': 'p1_elo_bestPoints', 'elo_rank_loser': 'p2_elo_rank', 'elo_points_loser': 'p2_elo_points',
     'elo_bestRank_loser': 'p2_elo_bestRank', 'elo_bestPoints_loser': 'p2_elo_bestPoints'}
RENAME_OPTIONAL_DICT =\
    {'loser_seed': 'p2_seed', 'loser_name': 'p2_name', 'loser_ioc': 'p2_ioc', 'winner_name': 'p1_name',
     'winner_ioc': 'p1_ioc', 'winner_entry': 'p1_entry', 'loser_entry': 'p2_entry', 'winner_seed': 'p1_seed',
     'B365W': 'p1_B365', 'B365L': 'p2_B365', 'CBW': 'p1_CB', 'CBL': 'p2_CB', 'GBW': 'p1_GB', 'GBL': 'p2_GB',
     'IWW': 'p1_IW', 'IWL': 'p2_IW', 'SBW': 'p1_SB', 'SBL': 'p2_SB', 'UBW': 'p1_UB', 'UBL': 'p2_UB', 'SJW': 'p1_SJ',
     'SJL': 'p2_SJ', 'EXW': 'p1_EX', 'EXL': 'p2_EX', 'WPts': 'p1_Pts', 'LPts': 'p2_Pts', 'PSW': 'p1_PS', 'PSL': 'p2_PS',
     'LBW': 'p1_LB', 'LBL': 'p2_LB', 'MaxW': 'p1_Max', 'MaxL': 'p2_Max', 'AvgW': 'p1_Avg', 'AvgL': 'p2_Avg',
     'B&WW': 'p1_B&W', 'B&WL': 'p2_B&W'}


def add_player_id_to_betting():
    """
    Add the winner and loser IDs for every match in betting odds data files.
    The IDs are saved in Data/relevant_players file, having also first name and last name columns.
    betting odds files have winner and loser last name and first letter of first name.
    The func fits the name from the betting odds data to the right player in relevant player table.
    The new tables with the IDs is saved to folder Data/betting_odds/ named by the year.
    """
    players = pd.read_csv('Data/relevant_players.csv')
    players['first_name'] = players['first_name'].str.lower()
    players['last_name'] = players['last_name'].str.lower()

    for year in range(2001, 2022):
        file_type = '.xls' if year < 2013 else '.xlsx'
        betting_odds = pd.read_excel('Data/betting_odds/' + str(year) + file_type)
        betting_odds['Winner'] = betting_odds['Winner'].str.lower()
        betting_odds['Loser'] = betting_odds['Loser'].str.lower()

        betting_odds['winner_id'] = betting_odds['Winner'].apply(
            lambda row: get_player_id_by_last_name_and_first_letter_of_name(row, players, year))
        betting_odds['loser_id'] = betting_odds['Loser'].apply(
            lambda row: get_player_id_by_last_name_and_first_letter_of_name(row, players, year))

        print(f' for {year}: num of missing winners id: {betting_odds.winner_id.isna().sum()}')
        print(f' for {year}: num of missing loser id: {betting_odds.loser_id.isna().sum()}')

        betting_odds.to_csv(f'Data/betting_odds/{year}.csv', index=False)


def get_player_id_by_last_name_and_first_letter_of_name(last_space_letter_dot: str, players: pd.DataFrame, year: int):
    """
    return player ID by find the player in relevant players, based on last name and first letter of first name.
    There are names which can't be specified by the format name provided, after analyzing all of this names,
    add them (manualy) to the dictionaries above.
    :param last_space_letter_dot: string represent player name in format: <last name> <first letter of first name>.
    :param players: table with full players name near to their ID
    :param year: the year of the file we are process - for debug
    :return: The ID of the player
    """
    last_space_letter_dot = last_space_letter_dot.strip()

    if last_space_letter_dot in PROBLEMATIC_NAMES.keys():
        return PROBLEMATIC_NAMES[last_space_letter_dot]
    duplicated_names_of_this_year = DICT_DUPLICATE_BY_YEAR[str(year)]

    if last_space_letter_dot in duplicated_names_of_this_year.keys():
        return duplicated_names_of_this_year[last_space_letter_dot]

    try:
        last_name, first_letter_of_name = last_space_letter_dot.rsplit(' ', 1)
        players_with_same_last_name = players[players.last_name == last_name]
        matches_players = players_with_same_last_name[players_with_same_last_name
            .first_name.str[0].isin([first_letter_of_name[0]])]

        if len(matches_players) > 1:
            print(f'{year}: {last_space_letter_dot} can\'t be recognize, have: {matches_players}')
            return None
        # Print for debug:
        """
        print(f'{last_space_letter_dot} -> {str(matches_players.last_name)} {str(matches_players.first_name)} '
              f'id: {int(matches_players.player_id)}')
        """
        return int(matches_players.player_id)

    except:
        print(f'{year}: {last_space_letter_dot} have other issue')
        return None


def add_player_id_to_elo():
    """
    Add the player ID for every player in elo ranking data files.
    The IDs are saved in Data/relevant_players file, having also first name and last name columns.
    elo ranking files have full name column.
    The func fits the name from the elo ranking data to the right player in relevant player table.
    The new tables with the IDs is saved to folder Data/elo_ranking/ named by the year.
    """
    players = pd.read_csv('Data/relevant_players.csv')
    players['first_name'] = players['first_name'].str.lower()
    players['last_name'] = players['last_name'].str.lower()
    players['full_name'] = players['first_name'] + ' ' + players['last_name']

    for year in range(2000, 2022):
        elo_rank = pd.read_csv(f'Data/elo_ranking/Rankings{year}.csv', encoding='ISO-8859-1')
        elo_rank['name'] = elo_rank['name'].str.lower()
        elo_rank['player_id'] = elo_rank['name'].apply(
            lambda row: get_player_id_by_full_name(row, players, year))
        print(f' for {year}: num of missing ids: {elo_rank.player_id.isna().sum()}')
        elo_rank.to_csv(f'Data/elo_ranking/{year}.csv', index=False)


def get_player_id_by_full_name(full_name: str, players: pd.DataFrame, year: int):
    """
    return player ID by find the player in relevant players, based on full name.
    There are names which can't be specified by the format name provided, (second names' ect),
    add them (manualy) to the dictionaries above.
    :param full_name: string represent the players' full name in format: <first name> <last name>
    :param players: table with full players name near to their ID
    :param year: the year of the file we are process - for debug
    :return: The ID of the player
    """
    if full_name in ELO_PROBLAMETIC_NAMES.keys():
        return ELO_PROBLAMETIC_NAMES[full_name]

    try:
        players_with_same_full_name = players[players.full_name == full_name]

        if len(players_with_same_full_name) > 1:
            print(f'{year}: {full_name} can\'t be recognize, have: {players_with_same_full_name}')
            return None

        return int(players_with_same_full_name.player_id)

    except:
        print(f'{year}: {full_name} hava other issue')
        return None


def create_csv_of_relevant_players():
    """
    create subtable from the original Data/atp_players.csv, which contains players who played in the 60's-90's.
    the players in the relevant players are ones that participated in matches from 2000.
    The matches are saved in Data/atp_matches/ folder.
    """
    players = pd.read_csv('Data/atp_players.csv')
    all_relevant_players_ids = set()

    for atp_match_file in os.listdir('Data/atp_matches/'):
        if atp_match_file.endswith(".csv"):
            atp_matches = pd.read_csv('Data/atp_matches/' + atp_match_file)
            all_relevant_players_ids = all_relevant_players_ids.union(set(atp_matches.winner_id.tolist()))
            all_relevant_players_ids = all_relevant_players_ids.union(set(atp_matches.loser_id.tolist()))
        else:
            continue

    print(f'relevant ids: {all_relevant_players_ids}')
    relevant_players = players[players.player_id.isin(all_relevant_players_ids)]
    print(relevant_players.info)
    relevant_players.to_csv('Data/relevant_players.csv', index=False)


def check_relevant_players():
    """
    Verify the relevant players table created as expected - for debug
    """
    relevant_players = pd.read_csv('Data/relevant_players.csv')
    relevant_players['full_name'] = relevant_players['first_name'] + ' ' + relevant_players['last_name']
    print(relevant_players.full_name.value_counts())


def add_elo_ranking_to_matches_by_year():
    """
    Add the elo ranking features to the matches data. elo features changed once a year, though all matches of a player
    in same year will get the same elo ranking value.
    the new tables will be saved at Data/elo_ranking/ by format matches{year}.csv
    """
    for year in range(2000, 2022):
        elo_rank = pd.read_csv(f'Data/elo_ranking/{year}.csv')
        matches = pd.read_csv(f'Data/atp_matches/atp_matches_{year}.csv')
        elo_rank.rename(columns={'rank': 'elo_rank'}, inplace=True)

        matches['elo_rank_winner'], matches['elo_points_winner'],  matches['elo_bestRank_winner'],\
            matches['elo_bestRankDate_winner'], matches['elo_rankDiff_winner'], matches['elo_pointsDiff_winner'],\
            matches['elo_bestPoints_winner'], = zip(*matches['winner_id'].apply(
                lambda row: get_elo_ranking_by_id(row, elo_rank)))

        matches['elo_rank_loser'], matches['elo_points_loser'], matches['elo_bestRank_loser'], \
            matches['elo_bestRankDate_loser'], matches['elo_rankDiff_loser'], matches['elo_pointsDiff_loser'], \
            matches['elo_bestPoints_loser'], = zip(*matches['loser_id'].apply(
                lambda row: get_elo_ranking_by_id(row, elo_rank)))

        print(f' for {year}: num of missing elo_rank_winner: {matches.elo_rank_winner.isna().sum()}')
        print(f' for {year}: num of missing elo_rank_loser: {matches.elo_rank_loser.isna().sum()}')

        matches.to_csv(f'Data/elo_ranking/matches{year}.csv', index=False)


def get_elo_ranking_by_id(player_id: int, elo_ranking: pd.DataFrame):
    """
    Get all features related to elo ranking of a specific player.
    :param player_id: player ID
    :param elo_ranking: ranking table of a specific year
    :return: tuple of all player's values of elo ranking
    """
    player_data = elo_ranking[elo_ranking.player_id == player_id]
    if player_data.empty:
        return None, None, None, None, None, None, None

    return int(player_data['elo_rank']), int(player_data['points']), int(player_data['bestRank']),\
        player_data['bestRankDate'], int(player_data['rankDiff']), int(player_data['pointsDiff']),\
        int(player_data['bestPoints'])


def add_betting_to_matches():
    """
    merge betting odds features to matches data. for every betting odds example, add column of the closest match in
    matches data (the dates are not equall for sure, because in matches data the dates are usually the Monday of the
    tournament week) and then join then by winnerID, loserID and date.
    :return: save the new merged table to csv file.
    """
    all_betting_oods_features = ['B365W', 'B365L', 'B&WW', 'B&WL', 'CBW', 'CBL']
    for year in range(2001, 2022):
        betting_oods = pd.read_csv(f'Data/betting_odds/{year}.csv')
        matches = pd.read_csv(f'Data/elo_ranking/matches{year}.csv')
        matches['tourney_date'] = matches['tourney_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
        all_uniques_dates = matches['tourney_date'].unique()
        exist_features = betting_oods.columns.values

        """
        for feature in all_betting_oods_features:
            if feature not in exist_features:
                betting_oods[feature] = ""
        """

        betting_oods['Date'] = betting_oods['Date'].apply(lambda x: pd.to_datetime(x))
        betting_oods['tourney_date'] = betting_oods['Date'].apply(lambda row: nearest(all_uniques_dates, row))


        betting_relevant_cols = betting_oods.drop(['ATP', 'Location', 'Tournament', 'Date', 'Series', 'Surface',
                                                   'Round', 'Best of', 'Winner', 'Loser', 'WRank', 'LRank', 'W1', 'L1',
                                                   'W2', 'L2', 'W3', 'L3', 'W4', 'L4', 'W5', 'L5', 'Wsets', 'Lsets',
                                                   'Comment'], axis='columns').copy()

        #betting_relevant_cols = betting_relevant_cols.drop('Date', axis='columns')
        result = pd.merge(matches, betting_relevant_cols, how="left",
                          on=['winner_id', 'loser_id', 'tourney_date'])
        result.to_csv(f'Data/final{year}.csv', index=False)


def nearest(items, pivot):
    """
    Find the nearest value. in use to find the nearest date.
    :param items: List of items to find the neatest from them.
    :param pivot: The value to find the nearest from.
    :return: The nearest value.
    """
    return min(items, key=lambda x: abs(x - pivot))


def built_all_data_before_processing_table():
    """
    Concat all years matches together to one big table.
    :return: save the new table to a csv file.
    """
    all_years = [pd.read_csv(f'Data/final{year}.csv') for year in range(2001, 2022)]
    all_years_without_process = pd.concat(all_years)
    all_years_without_process.to_csv('Data/all_years_without_process.csv')


def plot_missing_values_present_per_feature():
    dataset = pd.read_csv('Data/all_years_without_process.csv')
    dataset = dataset.rename(columns=RENAME_DICT)
    curr_col = dataset.columns.values

    for label in RENAME_OPTIONAL_DICT.keys():
        if label in curr_col:
            dataset = dataset.rename(columns={label: RENAME_OPTIONAL_DICT[label]})

    # static match data
    group1 = {'tourney_id', 'tourney_name', 'surface', 'draw_size', 'tourney_level', 'tourney_date', 'match_num',
              'best_of', 'round', 'Court'}
    # static player data
    group2 = {'p1_id', 'p1_seed', 'p1_entry', 'p1_name', 'p1_hand', 'p1_ht', 'p1_age', 'p1_ioc', 'p2_id', 'p2_seed',
              'p2_entry', 'p2_name', 'p2_hand', 'p2_ht', 'p2_age', 'p2_ioc',
              'p1_atp_rank', 'p1_atp_rank_points', 'p2_atp_rank', 'p2_atp_rank_points', 'p1_elo_rank', 'p1_elo_points',
              'p1_elo_bestRank', 'p1_elo_bestPoints', 'p2_elo_rank',
              'p2_elo_points', 'p2_elo_bestRank', 'p2_elo_bestPoints'}
    # players performance
    group3 = {'minutes', 'p1_ace', 'p1_df', 'p1_svpt', 'p1_1stIn', 'p1_1stWon', 'p1_2ndWon', 'p1_SvGms', 'p1_bpSaved',
              'p1_bpFaced', 'p2_ace', 'p2_df', 'p2_svpt', 'p2_1stIn', 'p2_1stWon', 'p2_2ndWon', 'p2_SvGms',
              'p2_bpSaved', 'p2_bpFaced'}
    # betting odds
    group4 = {'p1_CB', 'p2_CB', 'p1_GB', 'p2_GB', 'p1_IW', 'p2_IW', 'p1_SB', 'p2_SB', 'p1_B365', 'p2_B365', 'p1_B&W',
              'p2_B&W', 'p1_EX', 'p2_EX', 'p1_PS', 'p2_PS', 'p1_UB', 'p2_UB', 'p1_LB', 'p2_LB', 'p1_SJ',
              'p2_SJ'}
    # scores
    group5 = {'score'}

    groups = [group1, group2, group3, group4, group5]

    for group in groups:
        data = dataset[list(group)]
        percent_missing = data.isnull().sum() * 100 / len(data)
        missing_value_df = pd.DataFrame({'column_name': data.columns,
                                         'percent_missing': percent_missing})
        missing_value_df.plot.bar()
        plt.plot()

        pd.set_option('max_colwidth', 150)
        pd.set_option('max_rows', 150)
        print(missing_value_df)


def delete_irrelevant_features():
    """
    Delete irrelevant feature (explain for each of them in the report).
    :return: save the new table to csv.
    """
    features_to_delete = ['_merge', 'winner_seed', 'winner_entry', 'winner_name', 'loser_seed', 'loser_entry',
                          'loser_name', 'elo_bestRankDate_loser', 'elo_bestRankDate_winner', 'elo_rankDiff_winner',
                          'elo_rankDiff_loser', 'elo_pointsDiff_winner', 'elo_pointsDiff_loser', 'loser_ioc',
                          'winner_ioc']
    for year in range(2001, 2022):
        all_data = pd.read_csv(f'Data/final{year}.csv')
        filtered_data = all_data.drop(features_to_delete, axis='columns').copy()
        fix_players_with_2_ids(filtered_data)

        filtered_data.to_csv(f'Data/first_filter{year}.csv', index=False)


def fix_players_with_2_ids(data):
    """
    We found some players that have 2 IDs (but are the same one). so change the Ids to one ID.
    :param data: The Dataframe to change
    """
    data.loc[data.winner_id == 103863, 'winner_id'] = 103862
    data.loc[data.loser_id == 103863, 'loser_id'] = 103862

    data.loc[data.winner_id == 103921, 'winner_id'] = 103920
    data.loc[data.loser_id == 103921, 'loser_id'] = 103920

    data.loc[data.winner_id == 110685, 'winner_id'] = 104711
    data.loc[data.loser_id == 110685, 'loser_id'] = 104711

    data.loc[data.winner_id == 104624, 'winner_id'] = 104623
    data.loc[data.loser_id == 104624, 'loser_id'] = 104623


def add_scores_features():
    """
    Delete rows of matches without final result/technical win.
    Transform the score feature from format: a-b(c) d-e(f)... to 15 features named:
    p1_set{i}_score, p1_set{i}_score, set{i}_breakpoint_score
    :return: save the new table to csv file.
    """
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/first_filter{year}.csv')
        data = data[~data.score.str.contains("RET")]
        data = data[~data.score.str.contains("Played and unfinished")]
        data = data[~data.score.str.contains("W/O")]
        data = data[~data.score.str.contains("In Progress")]
        data = data[~data.score.str.contains("Played and abandoned")]
        data = data[~data.score.str.contains("Walkover")]
        data = data[~data.score.str.contains("DEF")]
        data = data[~data.score.str.contains("Def")]
        data = data[~data.score.str.contains("Unfinished")]
        data = data[~data.score.str.contains("Apr")]

        data['p1_set1_score'], data['p2_set1_score'], data['set1_breakpoint_score'], \
        data['p1_set2_score'], data['p2_set2_score'], data['set2_breakpoint_score'], \
        data['p1_set3_score'], data['p2_set3_score'], data['set3_breakpoint_score'], \
        data['p1_set4_score'], data['p2_set4_score'], data['set4_breakpoint_score'], \
        data['p1_set5_score'], data['p2_set5_score'], data['set5_breakpoint_score'] = zip(*data['score'].apply(
            lambda row: get_scores_by_sets(row, year)))

        data_with_scores = data.drop('score', axis='columns').copy()
        data_with_scores.to_csv(f'Data/first_filter{year}scores.csv', index=False)


def get_scores_by_sets(score_str, year):
    """
    Split the score string initial format to 15 separated values.
    a-b(c) d-e(f) g-h(i) j-k(l) m-n(o) --> [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o]
    a-b(c) d-e(f) g-h(i) --> [a, b, c, d, e, f, g, h, i, 0, 0, 0, 0, 0, 0]
    a-b d-e(f) g-h(i) j-k(l) m-n --> [a, b, 0, d, e, f, g, h, i, j, k, l, m, n, 0]
    :param score_str: String represent all match scores
    :param year: The year of the match, for debug
    :return: list of all scores
    """
    data_to_return = []
    sets_scores = score_str.split()
    for s_set in sets_scores:
        try:
            p1_score, rest = s_set.split('-')
            if '(' in rest:
                p2_score, rest = rest.split('(')
                brkpoint_score = rest[:-1]
            else:
                p2_score = rest
                brkpoint_score = 0
        except:
            values_to_add = 15 - len(data_to_return)
            data_to_return.extend([0] * values_to_add)
            print(year, score_str, data_to_return)
            return data_to_return
        data_to_return.append(p1_score)
        data_to_return.append(p2_score)
        data_to_return.append(brkpoint_score)
    values_to_add = 15 - len(data_to_return)
    data_to_return.extend([0] * values_to_add)
    return data_to_return


def misatch_p1_p2():
    """
    Random rows to swap the players data. change the labels from winner_id to p1_id and from loser_id to p2_id.
    Add a target label for p1_won -> if the row doesn't randomized put 1, else put 0.
    :return: save the new table to a csv file.
    """
    for year in range(2001, 2022):
        print(year)
        data = pd.read_csv(f'Data/p1p2_first_ver{year}.csv')
        features_to_swap = [label[3:] for label in data.columns.values if 'p2_' in label]
        for index, row in data.iterrows():
            if bool(random.getrandbits(1)):
                data.at[index, 'p1_won'] = 0
                for feature in features_to_swap:
                    try:
                        player1_val = 0 if pd.isnull(data.at[index, f'p1_{feature}']) else row[f'p1_{feature}']
                        player2_val = 0 if pd.isnull(data.at[index, f'p2_{feature}']) else row[f'p2_{feature}']
                        data.at[index, f'p1_{feature}'], data.at[index, f'p2_{feature}'] = player2_val,\
                                                                                                 player1_val
                    except:
                        print(year, feature, index)
                        print(player1_val, player2_val)
        data.to_csv(f'Data/p1p2_after_mismatch{year}.csv', index=False)


def change_winner_loset_to_p1_p2():
    """
    Rename all 'winner' labels to 'p1' and all 'loser' labels to 'p2'
    :return: save the new table to a csv file.
    """
    for year in range(2001, 2022):
        data = pd.read_csv(f'Data/first_filter{year}scores.csv')
        data['p1_won'] = 1
        data = data.rename(columns=RENAME_DICT)
        curr_col = data.columns.values
        for label in RENAME_OPTIONAL_DICT.keys():
            if label in curr_col:
                data = data.rename(columns={label: RENAME_OPTIONAL_DICT[label]})
        data.to_csv(f'Data/p1p2_first_ver{year}.csv', index=False)


def main():
    create_csv_of_relevant_players()
    check_relevant_players()
    add_player_id_to_betting()
    add_player_id_to_elo()
    add_elo_ranking_to_matches_by_year()
    built_all_data_before_processing_table()
    add_betting_to_matches()
    plot_missing_values_present_per_feature()
    delete_irrelevant_features()
    add_scores_features()
    change_winner_loset_to_p1_p2()
    misatch_p1_p2()




if __name__ == '__main__':
    main()
