import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor


from players_plays_per_year import players_plays_per_year
from create_data_rank_pred import get_player_age

TARGET = 'elo_rank'


def get_model_str(model, **kwargs):
    """
    Return The name of the model for printing titles.
    :param model: The model in tuning process
    :param kwargs: if there are parameters specify the model, the specific additional at kwargs['default_param']
    :return: string of the model name
    """
    if model is Ridge:
        return 'Ridge'
    elif model is LinearRegression:
        return 'Linear Regression'
    elif model is KernelRidge:
        return f"Kernel Ridge - {kwargs['default_param']}"
    elif model is SGDRegressor:
        return f"SGD Regressor - ({kwargs['default_param']})"
    elif model is RandomForestRegressor:
        return 'Random Forest'
    elif model is AdaBoostRegressor:
        return 'AdaBoost'


def get_model_params_str(model, **kwargs):
    """
    Return The name of the paremeters that in tuning for printing titles.
    :param model: The model in tuning process
    :param kwargs: if there are parameters specify the model, the specific additional at kwargs['default_param']
    :return: string/ strings with the name of the parameters of the model
    """
    if model is Ridge:
        return 'alpha'
    if model is LinearRegression:
        return None
    if model is KernelRidge:
        if kwargs['default_param'] == 'rbf':
            return 'alpha', 'gamma'
        else:
            return 'alpha', 'degree'
    elif model is SGDRegressor:
        if kwargs['default_param'] == 'eta0':
            return 'l_ratio', 'eta0'
        else:
            return 'l_ratio', 'alpha'
    elif model is RandomForestRegressor or model is AdaBoostRegressor:
        return 'n_estimators'


def get_args_for_regression_model(reg_model, parameter, parameter2=None, **kwargs):
    """
    Create dictionary of initial parameters by the regression model and parameter
    :param reg_model: The model we create the parameters dict for
    :param parameter: the value of parameter we are tuning, change each calling of this function
    :param parameter2: if needed, other parameter value we are tuning
    :param kwargs: if there are specific defualt params or other specifies
    :return: dict of the all relevant parameters of the model with their values
    """
    if reg_model is Ridge:
        return {'alpha': parameter, 'fit_intercept': True}
    elif reg_model is LinearRegression:
        return {}
    elif reg_model is SGDRegressor:
        if kwargs['default_param'] == 'eta0':
            return {'penalty': 'elasticnet', 'l1_ratio': parameter, 'eta0': parameter2}
        else:
            return {'penalty': 'elasticnet', 'l1_ratio': parameter, 'alpha': parameter2}
    elif reg_model is KernelRidge:
        if kwargs['default_param'] == 'rbf':
            return {'alpha': parameter, 'gamma': parameter2, 'kernel': kwargs['default_param']}
        else:
            return {'alpha': parameter, 'degree': parameter2, 'kernel': kwargs['default_param']}
    elif reg_model is RandomForestRegressor or reg_model is AdaBoostRegressor:
        return {'n_estimators': parameter}


def generate_validation(train, year, real_validation):
    """
    Generate "Fake" validation, based on the 4 last quartiles of the validation one. for all players plays in the year,
    the function finds the last 4 statistics each player have, and prepares an average for all statistics feature,
    updates the year, q=1 (4th quarter) and age of the player.
    :param train: train set, all players' statistics before the validation q
    :param year: the year of the validation
    :param real_validation: the real validation that we *can't* use because we don't know the future.
    :return: The generated validation to predict
    """
    # IDs of all players played this year - create a row of average statistics for all of them
    players_this_year = players_plays_per_year[str(year)]
    # Initiate empty DataFrame
    validation_created = pd.DataFrame()
    for player in players_this_year:
        # Find all earlier statistics of the player.
        player_id_norm = (player - 100644) / (210013 - 100644)
        id_epsilon = 0.000004  # numerical issues, need epsilon smaller then the gap between 2 normalized IDs
        data_knows_about_this_player = train[(player_id_norm - id_epsilon <= train['player_id']) & (train['player_id'] <= player_id_norm + id_epsilon)].copy()
        # Get only 4th last statistics
        forth_last_stats = data_knows_about_this_player.nlargest(4, ['year', 'q'])
        # Calculate average over the 4th rows
        mean_player_stats = forth_last_stats.mean(axis=0)
        # update known features
        mean_player_stats['year'] = (year % 100) / 21
        mean_player_stats['q'] = 1
        mean_player_stats['age'] = get_player_age(player, year, 1) / 44.276712328767125
        # Add the row to the validation DataFrame
        validation_created = validation_created.append(mean_player_stats, ignore_index=True)
    # Paste the real elo rank to the rows, for easy finding the mse, r2 score
    validation_created = validation_created.drop([TARGET], axis=1)
    real_target = real_validation[[TARGET, 'player_id']]
    final_validation = pd.merge(validation_created, real_target, how="right", on='player_id')
    final_validation.fillna(0, inplace=True)
    return final_validation


def generate_empty_players_stat(player):
    """
    Generate empty row for players that don't have history of statistics.
    :param player: Player ID
    :return: Dictionary of all necessary features on validation with value 0
    """
    return {'player_id': player, 'elo_rank': 0, 'bestRank': 0, 'bestRankDate': 0, '1stIn': 0, '1stWon': 0,
            '2ndWon': 0, 'SvGms': 0, 'ace': 0, 'bpFaced': 0, 'bpSaved': 0, 'df': 0, 'height': 0,
            'loss_players_with_lower_atp_rank': 0, 'num_of_participent': 0, 'num_of_participent_on_carpet': 0,
            'num_of_participent_on_clay': 0, 'num_of_participent_on_grass': 0, 'num_of_participent_on_hard': 0,
            'num_of_wins': 0, 'num_of_wins_on_carpet': 0, 'num_of_wins_on_clay': 0, 'num_of_wins_on_grass': 0,
            'num_of_wins_on_hard': 0, 'svpt': 0, 'wins_players_with_lower_atp_rank': 0}


def tune_hyperparameter(regression_model, data, p_range, dummy_valid_mse, dummy_valid_r2, y_log=False, x_log=False):
    """
    Tune regression models parameter. For each validation year, creates train set from the quartiles before the 4th of
    the year, generates validation in the way presented in generate_validation func, and for each parameter value in the
    range, fits and predicts, saves the mse and r2 scores.
    After going through all of the years and parameter values, Calculates the average mse and r2 for each parameter
    value (sum of all years' mses of this parameter value and divide by the amount of years in validation).
    Plots graphs of averages mse, r2. Adds the average mse, r2 of the Dummy, to get perspective if the results are good.
    :param regression_model: The regression model to tune. mostly sklearn class, must have fit, predict methods
    :param data: all data about quarters player statistics
    :param p_range: range of the parameter to tune values
    :param dummy_valid_mse: average validation mse - for plotting on the graph to get perspective
    :param dummy_valid_r2: average validation r2 - for plotting on the graph to get perspective
    :param y_log: True if the validation results is log-range, for plot suitable graph
    :param x_log:True if the parameter range is log-range, for plot suitable graph
    :return: parameter value with the lowest validation mse, parameter value with the highest validation r2
    """
    # Initial list to save results. The inner lists will contain the results of all years of parameter value.
    train_errors = [[] for _ in p_range]
    validation_errors = [[] for _ in p_range]
    train_r2s = [[] for _ in p_range]
    validation_r2s = [[] for _ in p_range]

    min_mse = 1
    max_r2 = -2
    train_mse_best_param = 1
    train_r2_best_param = 0
    best_hyper_param_mse = None
    best_hyper_param_r2 = None

    years_of_test = [2003, 2008, 2012, 2016, 2020]
    # Calculate the mse, r2 for all parameter values for each year
    for year in range(2001, 2020):
        if year in years_of_test:
            continue
        # Train contain all quartiles before the 4th of the year in validation
        train = data[(data['year'] <= ((year % 100) / 21 + 0.03)) & (data['q'] < 1)]
        # Real validation have the real players statistics in the end of the 4th quarter. can't be use as validation
        # because we don't know the future. Will be use to compare the predicted result with the target column
        real_validation = data[
            ((year % 100) / 21 - 0.03 <= data['year']) & (data['year'] <= year % 100) / 21 + 0.03 & (data['q'] == 1)]
        # Generate "fake" validation by average of the last 4 statistics each player have
        validation = generate_validation(train, year, real_validation)
        # Find the mse, r2 for those train-validation for every value in the range
        for i, param in enumerate(p_range):
            # Initiate the model with the tested parameter value
            model = regression_model(**get_args_for_regression_model(regression_model, param))
            # Fit, predict and find mse, r2 scores
            mse_res, r2_res = get_train_validation_mse_r2(model, train, validation)
            # save all results for later analyzing
            train_mse, valid_mse, = mse_res
            train_r2, valid_r2 = r2_res
            train_errors[i].append(train_mse)
            validation_errors[i].append(valid_mse)
            train_r2s[i].append(train_r2)
            validation_r2s[i].append(valid_r2)
            # Middle print for debug
            print(f'for year {year} and param {param}: {train_mse=}, {valid_mse=}, {train_r2=}, {valid_r2=}')

    # After all years calculations, find the average of each parameter value
    validation_mse_results = [sum(mses) / len(mses) for mses in validation_errors]
    train_mse_results = [sum(mses) / len(mses) for mses in train_errors]
    validation_r2_results = [sum(r2s) / len(r2s) for r2s in validation_r2s]
    train_r2_results = [sum(r2s) / len(r2s) for r2s in train_r2s]

    # Find best parameter leads to lowest validation mse
    for i, v_mse in enumerate(validation_mse_results):
        if v_mse < min_mse:
            min_mse = v_mse
            train_mse_best_param = train_mse_results[i]
            best_hyper_param_mse = p_range[i]

    # Find best parameter leads to higher validation r2
    for i, v_r2 in enumerate(validation_r2_results):
        if v_r2 > max_r2:
            max_r2 = v_r2
            train_r2_best_param = train_r2_results[i]
            best_hyper_param_r2 = p_range[i]

    # Plot average validation mse by parameter value
    plt.plot(p_range, train_mse_results, ls="-", color='b', label="train", zorder=0)
    plt.scatter(p_range, train_mse_results, marker='.', s=25, c='b', zorder=2)
    plt.plot(p_range, validation_mse_results, ls="-", color='r', label="validation", zorder=0)
    plt.scatter(p_range, validation_mse_results, marker='.', s=25, c='r', zorder=2)
    plt.plot(p_range, [dummy_valid_mse] * len(p_range), ls="-", color='g', label="Dummy", zorder=0)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.legend()
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Hyperparameter ({})'.format(get_model_params_str(regression_model)))
    plt.suptitle(f'ELO Rank - {get_model_str(regression_model)} MSE by Hyperparameter', fontsize=14)
    plt.title(f'best {get_model_params_str(regression_model)} = {best_hyper_param_mse} with min validation mse = '
              f'{round(min_mse, 5)}', fontsize=10)
    plt.grid(color='c', linestyle='-', linewidth=0.2, zorder=-1)
    plt.show()
    print(f'for {get_model_str(regression_model)}: best param {best_hyper_param_mse} with validation mse: {min_mse}, '
          f'and train mse {train_mse_best_param}. ')

    # Plot average validation r2 by parameter value
    plt.plot(p_range, train_r2_results, ls="-", color='b', label="train", zorder=0)
    plt.scatter(p_range, train_r2_results, marker='.', s=25, c='b', zorder=2)
    plt.plot(p_range, validation_r2_results, ls="-", color='r', label="validation", zorder=0)
    plt.scatter(p_range, validation_r2_results, marker='.', s=25, c='r', zorder=2)
    plt.plot(p_range, [dummy_valid_r2] * len(p_range), ls="-", color='g', label="Dummy", zorder=0)
    if x_log:
        plt.xscale('log')
    if y_log:
        plt.yscale('log')
    plt.legend()
    plt.ylabel('R2 Score')
    plt.xlabel('Hyperparameter ({})'.format(get_model_params_str(regression_model)))
    plt.suptitle(f'ELO Rank - {get_model_str(regression_model)} R2 score by Hyperparameter', fontsize=14)
    plt.title(f'best {get_model_params_str(regression_model)} = {best_hyper_param_r2} with max validation r2 = '
            f'{round(max_r2, 5)}', fontsize=10)
    plt.grid(color='c', linestyle='-', linewidth=0.2, zorder=-1)
    plt.show()
    print(f'for {get_model_str(regression_model)}: best param {best_hyper_param_r2} with validation r2 score: {max_r2}, '
          f'and train r2 score {train_r2_best_param}. ')

    return best_hyper_param_mse, best_hyper_param_r2


def tune_two_hyperparameters_plot_heatmap(regression_model, data, p1_range, p2_range, default_param=None,
                                          log_scale=False):
    """
    Tune regression models parameters. For each validation year, creates train set from the quartiles before the 4th of
    the year, generates validation in the way presented in generate_validation func, and for each pair of parameters
    values in the ranges, fits and predicts, saves the mse and r2 scores.
    After going through all of the years and parameters pairs values, Calculates the average mse and r2 for each pair
    parameters values (sum of all years' mses of this pair values and divide by the amount of years in validation).
    Plots heatmap of averages mse, r2.
    :param regression_model: The regression model to tune. mostly sklearn class, must have fit, predict methods
    :param data: all data about quarters player statistics
    :param p1_range: range of the first parameter to tune values
    :param p2_range: range of the first parameter to tune values
    :param default_param: specify for the model, like name of kernel
    :param log_scale: True if the validation results is log-range, for plot suitable graph
    :return: pair parameters values with the lowest validation mse,
             pair parameters values with the highest validation r2
    """
    min_mse = 1
    max_r2 = -2
    best_params_mse = None
    best_params_r2 = None

    # Initiate dictionaries to save results. keys: validation years. values: 2D list of mse/r2 score for all parameters
    # values pairs
    years_of_test = [2003, 2008, 2012, 2016, 2020]
    all_mses = {year: [] for year in range(2010, 2021) if year not in years_of_test}
    all_r2s = {year: [] for year in range(2010, 2021) if year not in years_of_test}
    # Calculate the mse, r2 for all pairs parameters values for each year
    for year in range(2010, 2020):
        if year in years_of_test:
            continue
        # Create data to fit
        train = data[(data['year'] <= ((year % 100) / 21 + 0.03)) & (data['q'] < 1)]
        # Create real validation, the target column will be compare to the predicted results
        real_validation = data[
            ((year % 100) / 21 - 0.03 <= data['year']) & (data['year'] <= year % 100) / 21 + 0.03 & (data['q'] == 1)]
        # Generate validation to predict base on the last 4 player's statistics
        validation = generate_validation(train, year, real_validation)
        # Loop for every pair of values
        for p1 in p1_range:
            p2_mses = []
            p2_r2s = []
            for p2 in p2_range:
                # Initiate model this the tested values
                model = regression_model(**get_args_for_regression_model(regression_model, p1, p2,
                                                                         default_param=default_param))
                # Fit, predict and find mse, r2 scores
                mse_res, r2_res = get_train_validation_mse_r2(model, train, validation)
                train_mse, valid_mse, = mse_res
                train_r2, valid_r2 = r2_res
                print(f'for year {year} and parameters {p1=}, {p2=}, {default_param if default_param else ""}: '
                      f'{train_mse=}, {valid_mse=}, {train_r2=}, {valid_r2=}')
                # Save results for later
                p2_mses.append(valid_mse)
                p2_r2s.append(valid_r2)
            all_mses[year].append(p2_mses)
            all_r2s[year].append(p2_r2s)

    # Calculate the average validation mse, r2 for every pair of parameters values
    # Save in these lists all average results of each pair (will be plot as heatmap)
    mse_results = []
    r2_results = []

    for i, p1 in enumerate(p1_range):
        row_mse = []
        row_r2 = []
        for j, p2 in enumerate(p2_range):
            # Get all result with same parameters value fit but different years
            same_params_mse = [all_mses[year][i][j] for year in range(2010, 2021) if year not in years_of_test]
            same_params_r2 = [all_r2s[year][i][j] for year in range(2010, 2021) if year not in years_of_test]
            # Calculate the average of results through the years
            avg_mse = sum(same_params_mse) / len(same_params_mse)
            avg_r2 = sum(same_params_r2) / len(same_params_r2)
            # Find the best parameters values for mse perspective
            if avg_mse < min_mse:
                min_mse = avg_mse
                best_params_mse = p1, p2
            # Find the best parameters values for r2 perspective
            if avg_r2 > max_r2:
                max_r2 = avg_r2
                best_params_r2 = p1, p2
            row_mse.append(avg_mse)
            row_r2.append(avg_r2)
        mse_results.append(row_mse)
        r2_results.append(row_r2)

    print(f'MSE: Best params: {best_params_mse} with mse: {min_mse}')
    print(f'R2: Best params: {best_params_r2} with r2: {max_r2}')

    # Plot heatmaps
    norm = LogNorm() if log_scale else None
    g = sns.heatmap(mse_results, xticklabels=p2_range, yticklabels=p1_range, norm=norm)
    plt.title(f"{get_model_str(regression_model, default_param=default_param)} by hyperparameters "
              f"{get_model_params_str(regression_model, default_param=default_param)}\n"
              f"best MSE={round(min_mse, 5)}")
    y_lable, x_labal = get_model_params_str(regression_model, default_param=default_param)
    plt.ylabel(y_lable)
    plt.xlabel(x_labal)
    plt.show()

    g = sns.heatmap(r2_results, xticklabels=p2_range, yticklabels=p1_range, norm=norm)
    plt.title(f"{get_model_str(regression_model, default_param=default_param)} by hyperparameters "
              f"{get_model_params_str(regression_model, default_param=default_param)}\n"
              f"best R2 score={round(max_r2, 5)}")
    y_lable, x_labal = get_model_params_str(regression_model, default_param=default_param)
    plt.ylabel(y_lable)
    plt.xlabel(x_labal)
    plt.show()

    return best_params_mse, best_params_r2


def get_train_validation_mse_r2(model, train, validation):
    """
    Fits the model on the train set, predicts on the validation set and calculate the mse, r2 scores
    :param model: Initiated model
    :param train: train set to fit the model
    :param validation: validation set to predict and compare the target column with
    :return: (mse_train_score, mse_validation_score), (r2_train_score, r2_validation_score)
    """
    X_train = train.drop([TARGET], axis=1)
    y_train = train[TARGET]
    X_val = validation.drop([TARGET], axis=1)
    y_val = validation[TARGET]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    mse_results = mean_squared_error(y_train, y_train_pred), mean_squared_error(y_val, y_val_pred)
    r2_results = r2_score(y_train, y_train_pred), r2_score(y_val, y_val_pred)
    return mse_results, r2_results


def print_weights_features_by_coef(train, regression_model, param, param2=None, default_param=None):
    """
    Print the weights of the features by incresing order, of the given model and parameters
    :param train: train set to fit the model
    :param regression_model: The regression model to fit
    :param param: first parameter for initiate the model
    :param param2: second optional parameter
    :param default_param: specify more parameters like kernel name
    """
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    model = regression_model(**get_args_for_regression_model(regression_model, param, param2,
                                                             default_param=default_param))
    model.fit(X, y)
    index_in_order_importance = np.argsort([abs(x) for x in model.coef_])
    for i in range(len(index_in_order_importance)):
        print(model.coef_[index_in_order_importance[i]],
              X.columns.values[index_in_order_importance[i]])


def main():
    data = pd.read_csv('Data/Quest4/all_years_after_min_max_elo.csv')
    # In this regression the 4th quarter of every year will be the validation for tuning parameters, instead of 5 years
    # that their 4 quarter will be the test for the tuned models.
    years_of_test = [2003, 2008, 2012, 2016, 2020]

    # Find Dummy regressor mean mse and r2
    mse_validations = []
    r2_validations = []
    for year in range(2010, 2020):
        if year in years_of_test:
            continue
        # Fit on all quartiles before the 4th of the year
        train = data[(data['year'] <= ((year % 100) / 21 + 0.03)) & (data['q'] < 1)]
        # The real players' statistics in the 4th quarter
        real_validation = data[
            ((year % 100) / 21 - 0.03 <= data['year']) & (data['year'] <= (year % 100) / 21 + 0.03) & (data['q'] == 1)]
        # Generate "fake" examples for validation, cause we don't know the future
        validation = generate_validation(train, year, real_validation)  # explain how at generate_validation func
        # Fit and predict Dummy regressor
        mse_res, r2_res = get_train_validation_mse_r2(DummyRegressor(), train, validation)
        dummy_train_mse, dummy_validation_mse = mse_res
        dummy_train_r2, dummy_validation_r2 = r2_res
        # save the results for the average later
        mse_validations.append(dummy_validation_mse)
        r2_validations.append(dummy_validation_r2)
        # for debug:
        # print(f'{year}: {dummy_validation_mse=}, {dummy_train_mse=}, {dummy_validation_r2=}, {dummy_train_r2=}')
    average_dummy_mse = sum(mse_validations) / len(mse_validations)
    average_dummy_r2 = sum(r2_validations) / len(r2_validations)
    print(f'avarege dummy regressor: {average_dummy_mse}, {average_dummy_r2}')

    #average_dummy_mse = 0.12891973410337926
    #average_dummy_r2 = -0.6610280962482579

    print('***RIDGE**')
    best_ridge_alpha_mse, best_ridge_alpha_r2 = tune_hyperparameter(Ridge, data, np.logspace(-5, 8, num=14),
                                                                    average_dummy_mse, average_dummy_r2, x_log=True)
    print('weights for best mse params:')
    print_weights_features_by_coef(data, Ridge, best_ridge_alpha_mse)
    print('weights for best r2 params:')
    print_weights_features_by_coef(data, Ridge, best_ridge_alpha_r2)

    print('\n\n\n***LINEAR REGRESSION***')
    tune_hyperparameter(LinearRegression, data, range(1), average_dummy_mse, average_dummy_r2)
    print_weights_features_by_coef(data, LinearRegression, 1)

    print('\n\n\n***SGD ALPHA***')
    best_mse_params, best_r2_params = tune_two_hyperparameters_plot_heatmap(
        SGDRegressor, data, [round(x, 1) for x in np.linspace(0, 1, 11)], np.logspace(-5, 5, num=11),
        default_param='alpha')
    best_sgd_lratio_mse, best_sgd_alpha_mse = best_mse_params
    best_sgd_lratio_r2, best_sgd_alpha_r2 = best_r2_params
    print('weights for best mse params:')
    print_weights_features_by_coef(data, SGDRegressor, best_sgd_lratio_mse, best_sgd_alpha_mse, default_param='alpha')
    print('weights for best r2 params:')
    print_weights_features_by_coef(data, SGDRegressor, best_sgd_lratio_r2, best_sgd_alpha_r2, default_param='alpha')

    print('\n\n\n***RIDGE KERNEL RBF***')
    best_rbf_alpha, best_rbf_gamma = tune_two_hyperparameters_plot_heatmap(
        KernelRidge, data, np.logspace(-4, 4, num=9), np.logspace(-4, 4, num=9), default_param='rbf')

    print('\n\n\n***RIDGE KERNEL POLY***')
    best_degree_alpha, best_degree_gamma = tune_two_hyperparameters_plot_heatmap(
        KernelRidge, data, np.logspace(-4, 4, num=9), range(2, 6), default_param='poly')

    print('\n\n\n''***Random Forest**')
    best_rf_n = tune_hyperparameter(RandomForestRegressor, data, range(50, 650, 50), average_dummy_mse,
                                    average_dummy_r2)

    print('\n\n\n***Adaboost**')
    best_adboost_n = tune_hyperparameter(AdaBoostRegressor, data, range(50, 650, 50), average_dummy_mse,
                                         average_dummy_r2)


if __name__ == '__main__':
    main()