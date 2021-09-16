import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC, SVC


TARGET = 'p1_won'
MODELS_STR = {KNeighborsClassifier: 'KNN',
              DecisionTreeClassifier: 'DecisionTree',
              LinearSVC: 'LinearSVC'}
MODELS_PARAM_STR = {KNeighborsClassifier: 'n_neighbors',
                    DecisionTreeClassifier: 'max_depth',
                    LinearSVC: 'C'}


def get_args_for_classification_model(classification_model, parameter):
    if classification_model is KNeighborsClassifier:
        return {'n_neighbors': parameter}
    elif classification_model is DecisionTreeClassifier:
        return {'criterion': "entropy",
                'max_depth': parameter}
    elif classification_model is LinearSVC:
        return {'C': parameter,
                'max_iter': 5000}


def model_prediction(classification_model, hyperparameter_range, train, subtitle=None, is_log_range=False):
    results_validation = []
    results_train = []
    best_hyperparameter_validation = 0
    best_hyperparameter_train = 0

    best_accuracy_validation = 0
    best_accuracy_train = 0

    X = train.drop([TARGET, 'tourney_date'], axis=1)
    y = train[TARGET]

    for i in hyperparameter_range:
        print(classification_model, i)

        kf1 = KFold(n_splits=5, shuffle=True)
        train_acc_kf = []
        valid_acc_kf = []
        for train_index, validation_index in kf1.split(train):

            X_train, X_validation = np.array(X)[train_index], np.array(X)[validation_index]
            y_train, y_validation = np.array(y)[train_index], np.array(y)[validation_index]

            model = classification_model(**get_args_for_classification_model(classification_model, i))
            model.fit(X_train, y_train)

            accuracy_validation = model.score(X_validation, y_validation)
            accuracy_train = model.score(X_train, y_train)

            train_acc_kf.append(accuracy_train)
            valid_acc_kf.append(accuracy_validation)

        results_validation.append(sum(valid_acc_kf) / len(valid_acc_kf))
        results_train.append(sum(train_acc_kf) / len(train_acc_kf))

        if sum(valid_acc_kf) / len(valid_acc_kf) > best_accuracy_validation:
            best_accuracy_validation = sum(valid_acc_kf) / len(valid_acc_kf)
            best_hyperparameter_validation = i

        if sum(train_acc_kf) / len(train_acc_kf) > best_accuracy_train:
            best_accuracy_train = sum(train_acc_kf) / len(train_acc_kf)
            best_hyperparameter_train = i

    print("{}: Validation Best Hyperparameter:{} with acc:{}".format(MODELS_STR[classification_model],
                                                                     best_hyperparameter_validation,
                                                                     best_accuracy_validation))
    print("{}: Train Best Hyperparameter:{} with acc:{}".format(MODELS_STR[classification_model],
                                                                best_hyperparameter_train,
                                                                best_accuracy_train))

    plt.plot(hyperparameter_range, results_validation, ls="-", color='r', label="validation", zorder=0)
    plt.scatter(hyperparameter_range, results_validation, marker='.', s=25, c='r', zorder=2)
    plt.plot(hyperparameter_range, results_train, ls="-", color='b', label="train", zorder=0)
    plt.scatter(hyperparameter_range, results_train, marker='.', s=25, c='b', zorder=2)
    if is_log_range:
        plt.xscale('log')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Hyperparameter ({})'.format(MODELS_PARAM_STR[classification_model]))
    if subtitle:
        plt.suptitle("{} Accuracy by Hyperparameter".format(MODELS_STR[classification_model]), fontsize=14)
        plt.title(subtitle, fontsize=10)
    else:
        plt.title("{} Accuracy by Hyperparameter".format(MODELS_STR[classification_model]))
    plt.grid(color='c', linestyle='-', linewidth=0.2, zorder=-1)
    plt.show()

    return best_hyperparameter_train, best_hyperparameter_validation


def plot_dt_risk_with_best_depth(train, param, data_title):
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    model = DecisionTreeClassifier(criterion="entropy", max_depth=param)
    model.fit(X, y)
    index_in_order_importance = np.argsort(model.feature_importances_)
    for i in range(len(index_in_order_importance)):
        print(model.feature_importances_[index_in_order_importance[i]], X.columns.values[index_in_order_importance[i]])
    plt.figure(figsize=(40, 25))
    plot_tree(model, feature_names=X.columns.values, filled=True, fontsize=8)
    plt.title(f'Decision tree for {data_title} with max depth {param}')
    plt.show()


def print_linear_svm_weights_features(train, param):
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    svm_model = LinearSVC(**get_args_for_classification_model(LinearSVC, param))
    svm_model.fit(X, y)
    print(svm_model.coef_[3])
    index_in_order_importance = np.argsort([abs(x) for x in svm_model.coef_[3]])
    for i in range(len(index_in_order_importance)):
        print(svm_model.coef_[3][index_in_order_importance[i]],
              X.columns.values[index_in_order_importance[i]])


def main():

    all_years_train = pd.read_csv('Data/train/all_years_with_date.csv')

    train_without_scores = all_years_train.drop([x for x in all_years_train.columns.values if 'set' in x], axis=1)

    static_feature_match = ['draw_size', 'tourney_date', 'p1_id', 'p1_hand', 'p1_ht', 'p1_age', 'p2_id', 'p2_hand',
                            'p2_ht', 'p2_age', 'best_of', 'p1_atp_rank', 'p1_atp_rank_points', 'p2_atp_rank',
                            'p2_atp_rank_points', 'p1_elo_rank', 'p1_elo_bestRank', 'p2_elo_rank', 'p2_elo_bestRank',
                            'p1_won', 'carpet', 'clay', 'grass', 'hard', 'masters_1000s', 'grand_slams',
                            'other_tour-level', 'challengers', 'satellites_ITFs', 'tour_finals', 'davis_cup', 'f_round',
                            'qf_round', 'r128_round', 'r16_round', 'r32_round', 'r64_round', 'rr_round', 'sf_round']
    train_static_match_data = all_years_train[static_feature_match]

    data_list_to_fit = [all_years_train, train_without_scores, train_static_match_data]
    data_title = ['all data', 'data without scores of sets', 'only static data of matches']

    # KNN tuned: all-data, without score data, only static data
    """
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation = model_prediction(KNeighborsClassifier, range(1, 101, 2), train_data,
                                                         subtitle=train_title)
        print(f'KNN for {train_title}: \n\t '
              f'Best train k_neigh is: {param_train}, Best validation k_neigh is: {param_validation}')
    
    """
    print()
    # DT tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation = model_prediction(DecisionTreeClassifier, range(1, 101), train_data,
                                                         subtitle=train_title)
        print(f'DT for {train_title}: \n\t '
              f'Best train max_depth is: {param_train}, Best validation max_depth is: {param_validation}')
        plot_dt_risk_with_best_depth(train_data, param_validation, train_title)

    print()
    # LinearSVM tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation = model_prediction(LinearSVC, np.logspace(-3, 3, num=100), train_data, is_log_range=True,
                                                         subtitle=train_title)
        print(f'LinearSVM for {train_title}: \n\t '
              f'Best train C is: {param_train}, Best validation C is: {param_validation}')
        print_linear_svm_weights_features(train_data, param_validation)


if __name__ == '__main__':
    main()