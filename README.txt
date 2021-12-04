Welcome to our AI project - Tennis Predictor! 

All python files: execute by running the files in the folder that has the folders Data and files. command: $python <filename>.py
Data of this project:
	- Files in 'Data' and 'Data/atp_matches' folder are from: https://github.com/JeffSackmann
	- Files in 'Data/betting_odds' folder are from: http://www.tennis-data.co.uk/alldata.php
	- Files in Data/elo_ranking' folder are from: https://www.ultimatetennisstatistics.com/eloRatings


/*** User interface for questions 1-3 ***/
To execute the interface without creating the data/models (on python 3.7.8 only), command:
$ python our_tennis_predictor.py
If the python version is different, run the fits_models_for_system.py script before.

Files explanation of this part:
creates_files_for_system.py - Creates necessary tables files for the system. 
	Uses files: 
		- Data/: atp_rankings_00s, atp_rankings_10s,
		- Data/relevant_players.csv,
		- Data/elo_ranking/: {year}.csv
		The first is from the data we collected, the other can be created by built_data and data_processing scripts
fits_models_for_system.py - fits 3 models (for all questions) and save them at files/fitted_models.
	Uses files (supplied in the project):
		- Data/train/all_years_final_normal_id.csv,
		- Data/train/train_without_scores.csv,
		- Data/train/train_static_match_data.csv,
		- Data/test/test_all_years_final.csv,
		- Data/test/test_without_scores.csv,
		- Data/test/test_static_match_data.csv
	the file can be created by build_data and data_processing scripts
our_tennis_predictor.py - Our user interface. by running this file ($python our_tennis_predictor.py)
	the user interface will be open, just answer the question and get a result.
	By default, the system saves the fitted models in a file that fits python version 3.7.8, so,
	for running with other versions, please run first fits_models_for_system.py
	Uses files in folder 'files', and system_utils.py file.
system_utils.py - contains helper features for creating suitable data of player by name and date. this file can't execute.


/*** experiment part for questions 1-3 ***/
Files explanation of this part:
built_data.py - for each year, join 3 data tables (matches, elo ranking, betting odds) to one table that contains all features. 
		Use files in folders: Data, Data/atp_matches, Data/betting_odds, Data/elo_ranking
data_processing.py - 
	* Splits the data to train and test
	* Drops correlative features
	* Applies normalization, one hot encoding
	* Fills missing data
	* Transform unique faetures values(dates).
	* Save final train and test big tables.
	Use files that were created by data_processing script.
tune_parameters.py - Tunes ranges of values of classification models and plots suitble graphs:
	* KNN
	* Desicion tree
	* SVM
	* SVM - poly kernel
	* SVM - rbf kernel
	* Random forest
	* Adaboost
	* Preceptron
	* MLP
	Uses files: 
		- Data/train/all_years_final_normal_id.csv,
		- Data/train/train_without_scores.csv,
		- Data/train/train_static_match_data.csv
	that supplied in the project, and can be created by data_processing script
test_accurcy.py - Finds the accuracy on the test set for each tuned model.
	Uses files: 
		- Data/train/all_years_final_normal_id.csv,
		- Data/train/train_without_scores.csv,
		- Data/train/train_static_match_data.csv,
		- Data/test/test_all_years_final.csv,
		- Data/test/test_without_scores.csv,
		- Data/test/test_static_match_data.csv
	that supplied in the project, and can be created by data_processing script


/*** experiment part for question 4 ***/
Files explanation of this part:
create_data_rank_pred.py - Creates the statistics tables by quartiles per player. Creates a table of all matches sorted by date for the second part of Q4.
	Uses files in folders: Data/elo_ranking/, Data/atp_matches/
tune_parameters_q4.py - Tunes ranges of values of classification models and plots suitble graphs:
	* DummyRegressor
	* Ridge
	* SGDRegressor
	* KernelRidge- poly kernel
	* KernelRidge- rbf kernel
	* Random forest
	* Adaboost
	Uses files: 
		- Data/Quest4/all_years_after_min_max_elo.csv
	that supplied in the project, and can be created by create_data_rank_pred script
calculate_rank.py - Calculates players' rank for each quarter based on predicting matches results of this quarter.
	Plots histograms of the differences between the real and predicted rank.
	Uses files: 
		- Data/Quest4/all_matches_static_data.csv
		- Data/Quest4/{year}q{q}.csv
	that supplied in the project, both can be created by create_data_rank_pred script
