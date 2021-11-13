import warnings
import pandas as pd
import numpy as np
import datetime
import math
import joblib
from system_utils import Player
from pandas.core.common import SettingWithCopyWarning


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


SURFACE_DICT = {'1': 'Carpet', '2': 'Clay', '3': 'Grass', '4': 'Hard'}
TOURNEY_LEVEL_DICT = {'1': 'G', '2': 'M', '3': 'C',  '4': 'S', '5': 'D', '6': 'A', '7': 'F'}
ROUND_DICT = {'1': 'F', '2': 'SF', '3': 'QF', '4': 'R16', '5': 'R32', '6': 'R64', '7': 'R128', '8': 'RR'}


class System:
    def __init__(self):
        # Load helper tables
        self.players_list = pd.read_csv('files/players_data.csv')
        self.ranks_list = pd.read_csv('files/ranks.csv')
        self.heights_list = pd.read_csv('files/heights.csv')
        # Initiate empty dictionary that will contain the values of the processed input of the user
        self.data_dict = dict()
        # Welcoming message
        print('Welcome to tennis match predictor by Linoy and Shachar!')
        # Get question to predict from the user
        question = self.choose_a_question()
        # For the chosen question, get the input values for all features of this question
        if question == '1':
            print('You chose the first question! Let\'s start')
            self.question_1_input()
        elif question == '2':
            print('You chose the second question! Let\'s start')
            self.question_2_input()
        elif question == '3':
            print('You chose the third question! Let\'s start')
            self.question_3_input()
        # Create DataFrame row from all data collected.
        # The self.data_dict items format: 'feature_name': [value from the user]
        self.df_to_predict = pd.DataFrame.from_dict(self.data_dict)
        # Process the values (normalize, one-hot-encoding, ext.)
        self.prepare_data_to_predict()
        # Predict by the saved fitted model and print the result
        self.predict_winner(question)

    def choose_a_question(self):
        inp = input('Choose question to predict (1-3), for explanations of the questions, press h\n')
        if inp in ['1', '2', '3']:
            return inp
        elif inp == 'h':
            pass
            # self.print_explanation()
        else:
            print("You must choose between 1, 2 or 3. or press h for explenations.")
            return self.choose_a_question()

    def question_1_input(self):
        self.static_match_data_input()
        self.game_performances_input()
        self.betting_odds_input()
        self.scores_of_sets_input()

    def question_2_input(self):
        self.static_match_data_input()
        self.game_performances_input()
        self.betting_odds_input()

    def question_3_input(self):
        self.static_match_data_input()

    def static_match_data_input(self):
        self.p1_name = self.get_player_name('first')
        self.p2_name = self.get_player_name('second')
        self.data_dict['tourney_date'] = [self.get_match_date()]  # Put in "list" for future process
        # Fill self,data_dict with static data by player's name + date: player id, hand, age, height, ranks at match date
        self.player1 = Player(self.players_list, self.p1_name, self.data_dict, self.ranks_list, self.heights_list, 1)
        self.player2 = Player(self.players_list, self.p2_name, self.data_dict, self.ranks_list, self.heights_list, 2)
        # Fill all static features, validate in each function the input fit to the feature
        self.data_dict['best_of'] = [self.get_best_of()]
        self.data_dict['draw_size'] = [self.get_draw_size()]
        self.data_dict['surface'] = [self.get_surface()]
        self.data_dict['tourney_level'] = [self.get_turney_level()]
        self.data_dict['round'] = [self.get_round()]

    def get_player_name(self, order):
        """
        Get player name from user in format: <First name> <Second name>.
        The function help to the user, when typing first letters of the name and press enter, a list of names starts
        with these letters is printed. If the list contains only one name, this name will be the chosen one.
        If the name doesn't exist, print message.
        :param order: 'first' if typing the name of the first player, else 'second' (for printing)
        :return: The name of the chosen player.
        """
        players_names = self.players_list['full_name']
        print(f'Enter name of the {order} player: <First_name> <Last_name> (press Enter for autocompleter)')
        while True:
            name = input('')
            players_with_this_start = [player for player in players_names if player and player.lower().startswith(name.lower())]
            if len(players_with_this_start) == 1:
                return players_with_this_start[0]
            elif len(players_with_this_start) > 1:
                list_of_names = '\t'.join([f'*{p_name}' for p_name in players_with_this_start])
                print(f'Possible players: {list_of_names}')
            else:
                print('Don\'t have player with this name, try other one')

    def get_draw_size(self):
        draw_size = input('Enter number of players in the draw: ')
        while True:
            if draw_size.isdigit():
                return 2 ** math.ceil(math.log2(int(draw_size)))
            else:
                draw_size = input('Invalid draw size. Enter number: ')

    def get_match_date(self):
        date_entry = input('Enter the match date in YYYY-MM-DD format: ')
        while True:
            try:
                year, month, day = map(int, date_entry.split('-'))
                self.year = year
                return datetime.date(year, month, day)

            except:
                date_entry = input('Invalid date was entered. Please enter a date in YYYY-MM-DD format: ')

    def get_best_of(self):
        best_of = input("The match is best of 3 or 5?: ")
        while True:
            if best_of == '3' or best_of == '5':
                return best_of
            best_of = input("Please enter one of the numbers 3/5: ")

    def get_surface(self):
        surface = input('Enter the surface type (1: Carpet, 2: Clay, 3: Grass, 4: Hard): ')
        while True:
            if surface.isdigit():
                if 1 <= int(surface) <= 4:
                    return SURFACE_DICT[surface]
            surface = input('Please enter a number between 1 to 4: ')

    def get_turney_level(self):
        level = input('Enter the tourney level:\n'
                        '1: Grand Slams,\n'
                        '2: Masters 1000s,\n'
                        '3: Challengers,\n'
                        '4: Satellites/ITFs,\n'
                        '5: Davis Cup,\n'
                        '6: other tour-level events,\n'
                        '7: Tour finals and other season-ending events\n')
        while True:
            if level.isdigit():
                if 1 <= int(level) <= 7:
                    return TOURNEY_LEVEL_DICT[level]
            level = input('Please enter a number between 1 to 7: ')

    def get_round(self):
        round_num = input('Enter the round of the match:\n'
                      '1: Final round,\n'
                      '2: Semi-Final round,\n'
                      '3: Qualification round,\n'
                      '4: Round of 16,\n'
                      '5: Round of 32,\n'
                      '6: Round of 64,\n'
                      '7: Round of 128\n'
                      '8: Round robin ("RR")')
        while True:
            if round_num.isdigit():
                if 1 <= int(round_num) <= 8:
                    return ROUND_DICT[round_num]
            round_num = input('Please enter a number between 1 to 8: ')

    def game_performances_input(self):
        # Fill all players' match performances features, validate in each function the input fit to the feature
        self.data_dict['minutes'] = [self.get_minutes()]
        self.data_dict['p1_ace'] = [self.get_number(player=1, feature_name='aces')]
        self.data_dict['p2_ace'] = [self.get_number(player=2, feature_name='aces')]
        self.data_dict['p1_df'] = [self.get_number(player=1, feature_name='doubles faults')]
        self.data_dict['p2_df'] = [self.get_number(player=2, feature_name='doubles faults')]
        self.data_dict['p1_svpt'] = [self.get_number(player=1, feature_name='serve points')]
        self.data_dict['p2_svpt'] = [self.get_number(player=2, feature_name='serve points')]
        self.data_dict['p1_1stWon'] = [self.get_number(player=1, feature_name='wons in first serve')]
        self.data_dict['p2_1stWon'] = [self.get_number(player=2, feature_name='wons in first serve')]
        self.data_dict['p1_2ndWon'] = [self.get_number(player=1, feature_name='wons in second serve')]
        self.data_dict['p2_2ndWon'] = [self.get_number(player=2, feature_name='wons in second serve')]
        self.data_dict['p1_svGms'] = [self.get_number(player=1, feature_name='serves in the game')]
        self.data_dict['p2_svGms'] = [self.get_number(player=2, feature_name='serves in the game')]
        self.data_dict['p1_bpSaved'] = [self.get_number(player=1, feature_name='breakpoint scores saved by player,')]
        self.data_dict['p2_bpSaved'] = [self.get_number(player=2, feature_name='breakpoint scores saved by player,')]
        self.data_dict['p1_bpFaced'] = [self.get_number(player=1, feature_name='breakpoint the player faced with,')]
        self.data_dict['p2_bpFaced'] = [self.get_number(player=2, feature_name='breakpoint the player faced with,')]

    def betting_odds_input(self):
        # Fill all players' betting odds features, validate in each function the input fit to the feature
        self.data_dict['p1_betting_odds'] = [self.get_betting_odds(player=1)]
        self.data_dict['p2_betting_odds'] = [self.get_betting_odds(player=2)]

    def scores_of_sets_input(self):
        # Fill all players' sets scores features, validate in each function the input fit to the feature
        self.data_dict['p1_set1_score'] = [self.get_score('set 1', player=1)]
        self.data_dict['p2_set1_score'] = [self.get_score('set 1', player=2)]
        self.data_dict['bp_set1_score'] = [self.get_bp_score(1)]
        self.data_dict['p1_set2_score'] = [self.get_score('set 2', player=1)]
        self.data_dict['p2_set2_score'] = [self.get_score('set 2', player=2)]
        self.data_dict['bp_set2_score'] = [self.get_bp_score(2)]
        self.data_dict['p1_set3_score'] = [self.get_score('set 3', player=1)]
        self.data_dict['p2_set3_score'] = [self.get_score('set 3', player=2)]
        self.data_dict['bp_set3_score'] = [self.get_bp_score(3)]
        self.data_dict['p1_set4_score'] = [self.get_score('set 4', player=1)] if self.data_dict['best_of'][0] == '5' else [0]
        self.data_dict['p2_set4_score'] = [self.get_score('set 4', player=2)] if self.data_dict['best_of'][0] == '5' else [0]
        self.data_dict['bp_set4_score'] = [self.get_bp_score(4)] if self.data_dict['best_of'][0] == '5' else [0]
        self.data_dict['p1_set5_score'] = [self.get_score('set 5', player=1)] if self.data_dict['best_of'][0] == '5' else [0]
        self.data_dict['p2_set5_score'] = [self.get_score('set 5', player=2)] if self.data_dict['best_of'][0] == '5' else [0]
        self.data_dict['bp_set5_score'] = [self.get_bp_score(5)] if self.data_dict['best_of'][0] == '5' else [0]

    def get_minutes(self):
        minutes = input('Enter how long was the game? (in minutes): ')
        while True:
            if minutes.isdigit():
                return int(minutes)
            minutes = input('Invalid input. please enter a number: ')

    def get_number(self, feature_name, player=None):
        required = input(f'Enter the number of {feature_name} of player {player} in the match: ')
        while True:
            if required.isdigit():
                return int(required)
            required = input('Invalid input. please enter a number: ')

    def get_score(self, category, player=None):
        score = input(f'Enter the score of {category} of player {player} in the match: ')
        while True:
            if score.isdigit():
                return int(score)
            score = input('Invalid input. please enter a number: ')

    def get_betting_odds(self, player):
        odds = input(f'Enter the betting oods for player {player} for the match: ')
        while True:
            try:
                return float(odds)

            except:
                odds = input('Invalid input. please enter a number (can be float): ')

    def get_bp_score(self, set_id):
        bp_score = input(f'Enter the score of breakpoint of set {set_id} in the match: ')
        while True:
            if bp_score.isdigit():
                return int(bp_score)

            bp_score = input('Invalid input. please enter a number: ')

    def prepare_data_to_predict(self):
        """
        Apply all processing data preformed on the train and test.
        """
        self.norm_all_features()
        self.apply_one_hot_encoding()
        self.fill_missing_test_data()
        self.transform_tests_dates()

    def norm_all_features(self):
        if 'p1_betting_odds' in self.data_dict.keys():
            # betting odds features don't exist in the third question
            self.norm_betting_odds()
        self.norm_z_score()
        self.norm_min_max()

    def norm_betting_odds(self):
        """
        Apply z-score normalization on the betting odds. Because every year has different scale, the std and mean of
        every year are the average std and mean of the stds and means of the betting odds houses in this year.
        :return: the DataFrame after the normalization
        """
        means_by_year = {'2001': 1.687341372347594, '2002': 1.588146130483476, '2003': 1.5609111296357259,
                         '2004': 1.816702734933353, '2005': 2.048949805269083, '2006': 2.18238294340514,
                         '2007': 2.2109252562620965, '2008': 2.154496365326963, '2009': 2.205774065589403,
                         '2010': 2.112811383647539, '2011': 2.4293449801144695, '2012': 2.5639174491414067,
                         '2013': 2.439744533296914, '2014': 2.420780576055154, '2015': 4.307968765096924,
                         '2016': 4.260707480788552, '2017': 2.3659395521773026, '2018': 2.1941511140897556,
                         '2019': 2.3393222844158355, '2020': 2.4213008636063, '2021': 2.3463881477184843}
        stds_by_year = {'2001': 1.1509622235945542, '2002': 1.2439617643850815, '2003': 1.4574682403419204,
                        '2004': 1.62783899592885, '2005': 2.10725387626062, '2006': 2.610042637363855,
                        '2007': 2.993271815581956, '2008': 2.6387494530487787, '2009': 2.6581874001078907,
                        '2010': 2.8185354034950447, '2011': 3.435316199440311, '2012': 4.298105695009052,
                        '2013': 3.317988023909592, '2014': 3.1964077200714827, '2015': 83.0252414213029,
                        '2016': 83.21491213954035, '2017': 2.593002780925955, '2018': 2.4048269616503193,
                        '2019': 2.7436167137227105, '2020': 3.42033454944029, '2021': 2.9977114440678734}
        year = str(self.year)
        self.df_to_predict['p2_betting_odds'] = (self.df_to_predict['p2_betting_odds'] - means_by_year[year]) / \
                                                stds_by_year[year]
        self.df_to_predict['p1_betting_odds'] = (self.df_to_predict['p1_betting_odds'] - means_by_year[year]) / \
                                                stds_by_year[year]
        return self.df_to_predict

    def norm_z_score(self):
        """
        Implement z-score normalization by the means and stds found in the data process.
        :return: the DataFrame after the normalization
        """
        mean_vals_dict = {'minutes': 107.46406599882145, 'p1_ace': 5.846042056589233, 'p2_ace': 5.832896620059452,
                          'p1_svpt': 76.89981283716834, 'p2_svpt': 76.94105471760432, 'p1_1stWon': 33.23939227127601,
                          'p1_2ndWon': 15.300605526808324, 'p2_1stWon': 33.23749862380271,
                          'p2_2ndWon': 15.315424419244742,
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
            if feature in self.data_dict.keys():
                self.df_to_predict[feature] = (self.df_to_predict[feature] - mean_vals_dict[feature]) / \
                                              std_vals_dict[feature]

        return self.df_to_predict

    def norm_min_max(self):
        """
        Implement min-max normalization by the mins and maxs found in the data process.
        :return: the DataFrame after the normalization
        """
        min_vals_dict = {'draw_size': 4, 'p1_df': 0.0, 'p1_bpSaved': 0.0, 'p1_bpFaced': 0.0, 'p2_df': 0.0,
                         'p2_bpSaved': 0.0, 'p2_bpFaced': 0.0, 'p1_atp_rank': 0.0, 'p1_atp_rank_points': 0.0,
                         'p2_atp_rank': 0.0, 'p2_atp_rank_points': 0.0, 'p1_elo_rank': 0.0, 'p1_elo_bestRank': 0.0,
                         'p2_elo_rank': 0.0, 'p2_elo_bestRank': 0.0, 'p1_id': 100644, 'p2_id': 100644}
        max_vals_dict = {'draw_size': 128, 'p1_df': 21.0, 'p1_bpSaved': 25.0, 'p1_bpFaced': 31.0, 'p2_df': 26.0,
                         'p2_bpSaved': 24.0, 'p2_bpFaced': 31.0, 'p1_atp_rank': 2147.0, 'p1_atp_rank_points': 16790.0,
                         'p2_atp_rank': 2159.0, 'p2_atp_rank_points': 16950.0, 'p1_elo_rank': 236.0,
                         'p1_elo_bestRank': 231.0, 'p2_elo_rank': 236.0, 'p2_elo_bestRank': 231.0, 'p1_id': 210013,
                         'p2_id': 210013}

        for feature in min_vals_dict.keys():
            if feature in self.data_dict.keys():
                self.df_to_predict[feature] = (self.df_to_predict[feature] - min_vals_dict[feature]) / (max_vals_dict[feature] - min_vals_dict[feature])

        return self.df_to_predict

    def apply_one_hot_encoding(self):
        """
        Implement one hot encoding on the categorical features
        :return: the DataFrame after the normalization
        """
        # transform surface to one-hot-encoding
        self.df_to_predict['carpet'] = np.where((self.df_to_predict['surface'] == 'Carpet'), 1, -1)
        self.df_to_predict['clay'] = np.where((self.df_to_predict['surface'] == 'Clay'), 1, -1)
        self.df_to_predict['grass'] = np.where((self.df_to_predict['surface'] == 'Grass'), 1, -1)
        self.df_to_predict['hard'] = np.where((self.df_to_predict['surface'] == 'Hard'), 1, -1)
        self.df_to_predict = self.df_to_predict.drop('surface', axis='columns').copy()

        # transform tourney_level to one-hot-encoding
        self.df_to_predict['masters_1000s'] = np.where((self.df_to_predict['tourney_level'] == 'M'), 1, -1)
        self.df_to_predict['grand_slams'] = np.where((self.df_to_predict['tourney_level'] == 'G'), 1, -1)
        self.df_to_predict['other_tour-level'] = np.where((self.df_to_predict['tourney_level'] == 'A'), 1, -1)
        self.df_to_predict['challengers'] = np.where((self.df_to_predict['tourney_level'] == 'C'), 1, -1)
        self.df_to_predict['satellites_ITFs'] = np.where((self.df_to_predict['tourney_level'] == 'S'), 1, -1)
        self.df_to_predict['tour_finals'] = np.where((self.df_to_predict['tourney_level'] == 'F'), 1, -1)
        self.df_to_predict['davis_cup'] = np.where((self.df_to_predict['tourney_level'] == 'D'), 1, -1)
        self.df_to_predict = self.df_to_predict.drop('tourney_level', axis='columns').copy()

        # transform pi_hand to binary number (R=1, L=-1)
        self.df_to_predict['p1_hand'] = np.where((self.df_to_predict['p1_hand'] == 'R'), 1, -1)
        self.df_to_predict['p2_hand'] = np.where((self.df_to_predict['p2_hand'] == 'R'), 1, -1)

        # transform best_of to binary number (3=1, 5=-1)
        self.df_to_predict['best_of'] = np.where((self.df_to_predict['best_of'] == 3), 1, -1)

        # transform round to one-hot-encoding
        self.df_to_predict['f_round'] = np.where((self.df_to_predict['round'] == 'F'), 1, -1)
        self.df_to_predict['qf_round'] = np.where((self.df_to_predict['round'] == 'QF'), 1, -1)
        self.df_to_predict['r128_round'] = np.where((self.df_to_predict['round'] == 'R128'), 1, -1)
        self.df_to_predict['r16_round'] = np.where((self.df_to_predict['round'] == 'R16'), 1, -1)
        self.df_to_predict['r32_round'] = np.where((self.df_to_predict['round'] == 'R32'), 1, -1)
        self.df_to_predict['r64_round'] = np.where((self.df_to_predict['round'] == 'R64'), 1, -1)
        self.df_to_predict['rr_round'] = np.where((self.df_to_predict['round'] == 'RR'), 1, -1)
        self.df_to_predict['sf_round'] = np.where((self.df_to_predict['round'] == 'SF'), 1, -1)
        self.df_to_predict = self.df_to_predict.drop('round', axis='columns').copy()

        return self.df_to_predict

    def transform_tests_dates(self):
        """
        Implement date transform like the transform in the data process
        :return: the DataFrame after the date transforn
        """
        first_date = pd.to_datetime('01/01/2001')
        self.df_to_predict['tourney_date'] = self.df_to_predict['tourney_date'].apply(lambda x: pd.to_datetime(x))
        self.df_to_predict['tourney_date'] = self.df_to_predict['tourney_date'].apply(lambda x: (x - first_date).days)
        self.df_to_predict['tourney_date'] = self.df_to_predict['tourney_date'] / 365

        return self.df_to_predict

    def fill_missing_test_data(self):
        """
        Fill missing values, by the means found in the data process.
        :return: the DataFrame after filling the missing values
        """
        means = {
            'p2_betting_odds': 0.013108076564071716, 'p2_atp_rank': 0.045668815063624385,
            'p2_2ndWon': 1.0013153162753806e-16, 'p2_elo_rank': 0.27468119002281255, 'minutes': -1.0718853291372165e-16,
            'p2_age': 2.0068707299812434e-16, 'p2_df': 0.11167946883018996, 'p2_ace': 3.003945948826142e-17,
            'p1_elo_rank': 0.26629257885810864, 'p2_elo_bestRank': 0.14148187741068718,
            'p2_bpSaved': 0.16625289001431245, 'p2_atp_rank_points': 0.07550748411717913,
            'p1_svpt': -1.251644145344226e-16, 'p2_SvGms': 1.451907208599302e-16, 'p2_1stWon': 4.0052612651015227e-17,
            'p1_elo_bestRank': 0.1365559144746962, 'p1_bpFaced': 0.2121396582768944,
            'p2_bpFaced': 0.21315822184655492, 'p1_SvGms': -1.6521702718543781e-16, 'p1_bpSaved': 0.1585476164262909,
            'p1_betting_odds': -0.00103945163478074, 'p1_atp_rank_points': 0.07481398897918069, 'p1_1stWon': 0.0,
            'p1_ht': 2.262620402022385e-16, 'p2_svpt': -2.5032882906884516e-17, 'p1_df': 0.1376459424461186,
            'p1_atp_rank': 0.046167961549591845, 'p2_ht': -3.7331437027798743e-16,
            'p1_ace': -1.2516441453442258e-17, 'p1_age': -8.122194752122179e-17, 'p1_2ndWon': -1.1890619380770144e-16}

        self.df_to_predict = self.df_to_predict.fillna(means)
        return self.df_to_predict

    def predict_winner(self, question):
        """
        Load the fitted model of the question, predict the DataFrame which saved in self.df_to_predict,
        and print the winner name.
        :param question: the number of question to predict
        """
        loaded_model = joblib.load(f'files/fitted_models/q{question}.sav')
        result = loaded_model.predict(self.df_to_predict)
        winner = self.p1_name if result[0] == 1 else self.p2_name
        print(f'Our model predicts the winner is \033[1m{winner}\033[0m!')


def main():
    System()


if __name__ == '__main__':
    main()
