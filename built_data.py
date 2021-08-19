import pandas as pd
import numpy as np
import os

PROBLEMATIC_NAMES = {'viloca j.a.': 102167, 'damm m.': 210013, 'lee h.t.': 102703, 'bahrouzyan o.': 103914,
                     'nadal-parera r.': 104745, 'andersen j.f.': 102107, 'vassallo-arguello m.': 103506,
                     'ivanov-smolensky k.': 103715, 'bogomolov jr.a.': 104166, 'elgin m.': 103862, 'silva f.': 106393,
                     'gambill j. m.': 102998, 'dell\'acqua m.': 103435, 'garcia-lopez g.': 104198, 'robredo r.': 103990,
                     'ramirez-hidalgo r.': 103105, 'bogomolov a.': 104166, 'o\'brien a.': 101711, 'goodall j.': 104623,
                     'al ghareeb m.': 103600, 'matos-gil i.': 104099, 'navarro-pastor i.': 103868, 'gruber k.': 103460,
                     'harper-griffith l.': 103834, 'herm-zahlava j.': 103564, 'garcia-sintes j.': 103365,
                     'al-alawi s.k.': 102956, 'bogomolov jr. a.': 104166, 'kunitcin i.': 103857, 'hantschek m.': 103082,
                     'ratiwatana s.': 103920, 'roger-vasselin e.': 104273, 'gallardo m.': 103746, 'gimeno d.': 104593,
                     'gimeno-traver d.': 104593, 'wang y. jr': 103817, 'querry s.': 105023, 'haider-mauer a.': 104890,
                     'sultan-khalfan a.': 102956, 'king-turner d.': 104368, 'van der dium a.': 105014,
                     'rascon t.': 101869, 'granollers-pujol m.': 104719, 'salva b.': 104831, 'luque d.': 104317,
                     'vicente m.': 102950, 'dev varman s.': 104500, 'fish a.': 103888, 'stepanek m.': 103285,
                     'fornell m.': 103924, 'guccione a.': 104589, 'chekov p.': 105189, 'trujillo g.': 103448,
                     'statham r.': 104907, 'jones g.d.': 104711, 'podlipnik h.': 105070, 'al-ghareeb m.': 103600,
                     'nader m.': 108996, 'lopez-jaen m.a.': 104091, 'o\'neal j.': 103076, 'de heart r.': 104281,
                     'ruevski p.': 104234, 'haji a.': 108993, 'riba-madrid p.': 105137, 'munoz-de la nava d.': 103926,
                     'del bonis f.': 105643, 'haider-maurer a.': 104890, 'ramos-vinolas a.': 105077, 'mecir m.': 105080,
                     'dolgopolov o.': 105238, 'granollers pujol g.': 105283, 'gutierrez-ferrol s.': 105300,
                     'carreno-busta p.': 105807, 'bautista r.': 105138, 'dutra da silva r.': 104297,
                     'granollers-pujol g.': 105283, 'cervantes i.': 105438, 'van d. merwe i.': 104292,
                     'brugues-davi a.': 104516, 'gomez-herrera c.': 105529, 'menendez-maceiras a.': 104629,
                     'deen heshaam a.': 106277, 'ali mutawa j.m.': 106325, 'zayed m. s.': 110476, 'silva f.f.': 106393,
                     'reyes-varela m.a.': 104960, 'estrella burgos v.': 103607, 'nedovyesov o.': 104873,
                     'artunedo martinavarro a.': 106238, 'galan d.e.': 123755, 'prashanth v.': 104822,
                     'ortega-olmedo r.': 105751, 'zayid m. s.': 122570, 'samper-montana j.': 105515,
                     'lopez-perez e.': 105782, 'del potro j. m.': 105223, 'o\'connell c.': 106331, 'alcaraz c.': 207989,
                     'alawadhi o.': 200525, 'hernandez-fernandez j.': 105497, 'auger-aliassime f.': 200000,
                     'varillas j.p.': 122669, 'o connell c.': 106331, 'galan d.': 123755, 'varillas j. p.': 122669,
                     'hernandez-fernandez j': 105497, 'meligeni rodrigues f': 200335, 'barrios m.': 144642,
                     'menezes j.': 110778}

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

def add_player_id_to_betting_2001():
    atp_matches_01 = pd.read_csv('Data/atp_matches/atp_matches_2001.csv')
    #betting_odds_01 = pd.read_excel('Data/betting_odds/2001.xls')
    #elo_rank_01 = pd.read_csv('Data/elo_ranking/Rankings2001.csv', encoding='ISO-8859-1')

    players = pd.read_csv('Data/relevant_players.csv')

    #atp_matches_01['tourney_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

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


def get_player_id_by_last_name_and_first_letter_of_name(last_space_letter_dot, players, year):
    last_space_letter_dot = last_space_letter_dot.strip()
    if last_space_letter_dot in PROBLEMATIC_NAMES.keys():
        return PROBLEMATIC_NAMES[last_space_letter_dot]
    duplicated_names_of_this_year = DICT_DUPLICATE_BY_YEAR[str(year)]
    if last_space_letter_dot in duplicated_names_of_this_year.keys():
        return duplicated_names_of_this_year[last_space_letter_dot]
    try:
        last_name, first_letter_of_name = last_space_letter_dot.rsplit(' ', 1)
        players_with_same_last_name = players[players.last_name == last_name]
        matches_players = players_with_same_last_name[players_with_same_last_name.first_name.str[0].isin([first_letter_of_name[0]])]
        if len(matches_players) > 1:
            print(f'{year}: {last_space_letter_dot} can\'t be recognize, have: {matches_players}')
            return None
        #print(f'{last_space_letter_dot} -> {str(matches_players.last_name)} {str(matches_players.first_name)} id: {int(matches_players.player_id)}')
        return int(matches_players.player_id)
    except:
        print(f'{year}: {last_space_letter_dot} hava other issue')
        return None


def create_csv_of_relevant_players():
    players = pd.read_csv('Data/atp_players.csv')
    all_relevant_players_ids = set()

    for atp_match_file in os.listdir('Data/atp_matches/'):
        if atp_match_file.endswith(".csv"):
            atp_matches = pd.read_csv('Data/atp_matches/' + atp_match_file)
            #ids_of_year = set(atp_matches.winner_id) + set(atp_matches.loser_id)
            #print(set(atp_matches.winner_id.tolist()))
            all_relevant_players_ids = all_relevant_players_ids.union(set(atp_matches.winner_id.tolist()))
            all_relevant_players_ids = all_relevant_players_ids.union(set(atp_matches.loser_id.tolist()))
            continue
        else:
            continue

    print(f'relevant ids: {all_relevant_players_ids}')
    relevant_players = players[players.player_id.isin(all_relevant_players_ids)]
    print(relevant_players.info)
    print()
    print(players.info)
    #relevant_players.to_csv('Data/relevant_players.csv', index=False)


def check_relevant_players():
    relevant_players = pd.read_csv('Data/relevant_players.csv')
    relevant_players['full_name'] = relevant_players['first_name'] + ' ' + relevant_players['last_name']
    print(relevant_players.full_name.value_counts())


def main():
    #create_csv_of_relevant_players()
    #check_relevant_players()
    add_player_id_to_betting_2001()


if __name__ == '__main__':
    main()
