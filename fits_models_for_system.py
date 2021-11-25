import pandas as pd
import joblib
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier


TARGET = 'p1_won'


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
    create_q1_model()
    create_q2_model()
    create_q3_model()


if __name__ == '__main__':
    main()