import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import KFold, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

# Target label
TARGET = 'p1_won'
# Dict to specify name for models
MODELS_STR = {KNeighborsClassifier: 'KNN',
              DecisionTreeClassifier: 'DecisionTree',
              LinearSVC: 'LinearSVC',
              RandomForestClassifier: 'RandomForestClassifier',
              AdaBoostClassifier: 'AdaBoost',
              MLPClassifier: 'Multi Layers Perceptron'}
# dict to specify name of parameters we tune for each classification model
MODELS_PARAM_STR = {KNeighborsClassifier: 'n_neighbors',
                    DecisionTreeClassifier: 'max_depth',
                    LinearSVC: 'C',
                    AdaBoostClassifier: 'n_estimators'}
# dict to specify name of parameters we tune for each kernel
KERNELS_PARAM_STR = {'poly': 'degree', 'rbf': 'gamma'}


def get_args_for_classification_model(classification_model, parameter):
    """
    Create dictionary of initial parameters by the classification model and parameter
    :param classification_model: The model we create the parameters dict for
    :param parameter: the value of parameter we are tuning and change each calling of this function
    :return: dictionary of initial parameters for the classification model
    """
    if classification_model is KNeighborsClassifier:
        return {'n_neighbors': parameter}
    elif classification_model is DecisionTreeClassifier:
        return {'criterion': "entropy",
                'max_depth': parameter}
    elif classification_model is LinearSVC:
        return {'C': parameter,
                'max_iter': 5000}
    elif classification_model is AdaBoostClassifier:
        return {'n_estimators': parameter}


def get_args_for_svc_with_kernel(kernel_type, c_param, kernel_param):
    """
    Create dictionary of initial parameters for svc with kernel
    :param kernel_type: the kernel to use
    :param c_param: the value of C parameter
    :param kernel_param: the value of the kernel parameter
    :return: dictionary of initial parameters for svc with kernel
    """
    if kernel_type == 'poly':
        return {'C': c_param, 'kernel': kernel_type, 'degree': kernel_param}
    return {'C': c_param, 'kernel': kernel_type, 'gamma': kernel_param}


def model_parameter_tuning(classification_model, hyperparameter_range, train, subtitle=None, is_log_range=False):
    """
    Tune classification model by cross-validation. for every value in the parameter range, split randomly the train for
    5 parts, fits and predict with the tested parameter value, calculate the average train and validation accuracy for
    this value.
    plot a graph of all averages accuracies, and print the best parameters in train aspect and validation aspect.
    :param classification_model: The classification model to tune. mostly sklearn class, must have fit, predict methods
    :param hyperparameter_range: range of the parameter to tune values
    :param train: DataFrame of the train data
    :param subtitle: the name of the train (by the 3 questions)
    :param is_log_range: True if the parameter range is log-range, for plot suitable graph
    :return: best train parameter, best validation parameter, best_model (fitted with the best validation hyperparameter)
    """
    results_validation = []
    results_train = []
    best_hyperparameter_validation = 0
    best_hyperparameter_train = 0
    best_accuracy_validation = 0
    best_accuracy_train = 0
    best_model = None

    X = train.drop([TARGET], axis=1)
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

        validation_avg = sum(valid_acc_kf) / len(valid_acc_kf)
        train_avg = sum(train_acc_kf) / len(train_acc_kf)
        print(f'validation: {valid_acc_kf} -> {validation_avg}')
        print(f'train: {train_acc_kf} -> {train_avg}')

        results_validation.append(validation_avg)
        results_train.append(train_avg)

        if validation_avg > best_accuracy_validation:
            best_accuracy_validation = validation_avg
            best_hyperparameter_validation = i
            best_model = model

        if train_avg > best_accuracy_train:
            best_accuracy_train = train_avg
            best_hyperparameter_train = i

    model_str = MODELS_STR[classification_model]
    print(f'{model_str}: Validation Best Hyperparameter:{best_hyperparameter_validation} with acc:{best_accuracy_validation}')
    print(f'{model_str}: Train Best Hyperparameter:{best_hyperparameter_train} with acc:{best_accuracy_train}')

    plt.plot(hyperparameter_range, results_validation, ls="-", color='r', label="validation", zorder=0)
    plt.scatter(hyperparameter_range, results_validation, marker='.', s=25, c='r', zorder=2)
    plt.plot(hyperparameter_range, results_train, ls="-", color='b', label="train", zorder=0)
    plt.scatter(hyperparameter_range, results_train, marker='.', s=25, c='b', zorder=2)
    if is_log_range:
        plt.xscale('log')
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel(f'Hyperparameter ({MODELS_PARAM_STR[classification_model]})')
    if subtitle:
        plt.suptitle(f'{model_str} Accuracy by Hyperparameter', fontsize=14)
        plt.title(subtitle, fontsize=10)
    else:
        plt.title(f'{model_str} Accuracy by Hyperparameter')
    plt.grid(color='c', linestyle='-', linewidth=0.2, zorder=-1)
    plt.show()

    return best_hyperparameter_train, best_hyperparameter_validation, best_model


def tune_svc_kernel_plot_heatmap(train, c_range, kernel, kernel_range, train_title=""):
    """
    Tune svc kernel by cross-validation. for every values of C and kernel parameters, split randomly the train for 5
    parts, fits and predict with the tested parameters values, calculate the average validation accuracy for this value.
    plot a heatmap of all averages accuracies, and print the best validation parameters.
    :param train: DataFrame of the train data
    :param c_range: range of values for tuning the kernel
    :param kernel: kind of kernel to tune
    :param kernel_range: range of values for tuning the C parameter
    :param train_title: the name of the train (by the 3 questions)
    """
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]

    best_acc = 0
    best_params = None
    all_accuracies = []

    for c in c_range:
        print(c)
        c_accuracies = []
        for i in kernel_range:

            kf1 = KFold(n_splits=5, shuffle=True)
            this_fold_accs = []
            for train_index, validation_index in kf1.split(train):
                X_train, X_validation = np.array(X)[train_index], np.array(X)[validation_index]
                y_train, y_validation = np.array(y)[train_index], np.array(y)[validation_index]

                model = SVC(**get_args_for_svc_with_kernel(kernel, c, i))
                model.fit(X_train, y_train)
                accuracy = model.score(X_validation, y_validation)
                this_fold_accs.append(accuracy)
            last_fold_avg_acc = sum(this_fold_accs) / len(this_fold_accs)
            c_accuracies.append(last_fold_avg_acc)
            if last_fold_avg_acc > best_acc:
                best_acc = last_fold_avg_acc
                best_params = (c, i)
            print(f'for kernel {kernel}, and params: C={c}, {KERNELS_PARAM_STR[kernel]}={i} '
                  f'with accuracy: {last_fold_avg_acc}')

        all_accuracies.append(c_accuracies)

    print(f'for kernel {kernel}, best params: C={best_params[0]}, '
          f'{KERNELS_PARAM_STR[kernel]}={best_params[1]} with accuracy: {best_acc}')
    print(all_accuracies)

    g = sns.heatmap(all_accuracies, xticklabels=kernel_range, yticklabels=c_range)
    plt.title(f'Validation accuracy of {kernel} kernel by hyperparameters C, '
              f'{KERNELS_PARAM_STR[kernel]}\n{train_title}')
    plt.ylabel('C')
    plt.xlabel(KERNELS_PARAM_STR[kernel])
    plt.show()


def tune_model_by_random_grid(classification_model, train, random_grid, train_title=""):
    """
    Tune classification model by cross validation, choose randomly parameters values from grid. The function does 50
    iterations of choosing random values from the grid, fiting and predicting the train by cross-validation of 5,
    calculating the validation accuracies and the average for each set of parameters values.
    :param classification_model: The classification model to tune. mostly sklearn class, must have fit, predict methods
    :param train: DataFrame of the train data
    :param random_grid: dict of parameters: 'p_name': [list of optional values] for each of them.
    :param train_title: the name of the train (by the 3 questions)
    :return: set of best parameters values that were chosen randomly
    """
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    model = classification_model()
    # Random search of parameters, using 5 fold cross validation,
    # search across 50 different combinations, and use all available cores
    model_random = \
        RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=50, cv=5, verbose=10, n_jobs=-1)
    # Fit the random search model
    model_random.fit(X, y)

    model_str = MODELS_STR[classification_model]
    print(f'{model_str} for {train_title}:\n'
          f'Validation Best Hyperparameters: {model_random.best_params_}\n'
          f'with acc:{model_random.best_score_}')

    return model_random.best_params_


def tune_model_by_grid(classification_model, train, grid, train_title=""):
    """
    Tune classification model by cross validation, testing all permutations of parameters values from a grid.
    The function create all permutations from the given grid, and for each permutation, fiting and predicting the train
    by cross-validation of 5, calculating the validation accuracies and the average for each set of parameters values.
    :param classification_model:  The classification model to tune. mostly sklearn class, must have fit, predict methods
    :param train: DataFrame of the train data
    :param grid: dict of parameters: 'p_name': [list of optional values] for each of them.
    :param train_title: the name of the train (by the 3 questions)
    :return: set of best parameters values from all permutation
    """
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    model = classification_model()
    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=5, n_jobs=-1, verbose=20)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    print(grid_search.best_estimator_)

    model_str = MODELS_STR[classification_model]
    print(f'{model_str} for {train_title}:\n'
          f'Validation Best Hyperparameters: {grid_search.best_params_}\n'
          f'with acc:{grid_search.best_score_}')

    return grid_search.best_params_


def tune_perceptron(train, train_title=""):
    """
    Tune perceptron by penalty and eta0 parameters, use repeated cross-validation, because the algorithm is stochastic.
    plot graph of the average validation accuracy for each penalty option. print the best accuracies for every penalty
    option.
    :param train: DataFrame of the train data
    :param train_title: the name of the train (by the 3 questions)
    """
    X = train.drop([TARGET], axis=1)
    y = train[TARGET]
    # define model evaluation method
    cv = RepeatedStratifiedKFold(n_splits=7, n_repeats=3, random_state=1)
    # define grid
    grid = dict()
    grid['eta0'] = np.logspace(-4, 2, num=16)
    grid['max_iter'] = [1500]
    loss_f = ['elasticnet', 'l2', 'l1', None]

    means_validation = dict()
    means_train = dict()

    for loss in loss_f:
        grid['penalty'] = [loss]
        if loss == 'elasticnet':
            grid['l1_ratio'] = [0.5]
        else:
            grid.pop("l1_ratio", None)
        print(f'grid: {grid}')
        # define search
        search = GridSearchCV(Perceptron(), grid, scoring='accuracy', cv=cv, n_jobs=-1, return_train_score=True)
        # perform the search
        results = search.fit(X, y)
        # summarize
        #print(f"{loss} validation: {results.cv_results_['mean_test_score']}")
        #print(f"{loss} train: {results.cv_results_['mean_train_score']}")
        print(f'penalty {loss} best acc: {results.best_score_} with params: {results.best_params_}')
        print_features_weights_by_coef(X, results.best_estimator_)
        # summarize all
        loss_str = loss if loss is not None else 'None'
        means_validation[loss_str] = results.cv_results_['mean_test_score']
        means_train[loss_str] = results.cv_results_['mean_train_score']

    plt.plot(grid['eta0'], means_validation['None'], ls="-", color='deeppink', label="panelty None", zorder=0)
    plt.scatter(grid['eta0'], means_validation['None'], marker='.', s=25, c='deeppink', zorder=2)

    plt.plot(grid['eta0'], means_validation['l2'], ls="-", color='orange', label="panelty l2", zorder=0)
    plt.scatter(grid['eta0'], means_validation['l2'], marker='.', s=25, c='orange', zorder=2)

    plt.plot(grid['eta0'], means_validation['l1'], ls="-", color='seagreen', label="panelty l1", zorder=0)
    plt.scatter(grid['eta0'], means_validation['l1'], marker='.', s=25, c='seagreen', zorder=2)

    plt.plot(grid['eta0'], means_validation['elasticnet'], ls="-", color='darkred', label="panelty elasticnet", zorder=0)
    plt.scatter(grid['eta0'], means_validation['elasticnet'], marker='.', s=25, c='darkred', zorder=2)

    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.ylabel('Validation Accuracy')
    plt.xlabel('Hyperparameter (eta0)')
    plt.suptitle("Perceptron Validation Accuracy by Hyperparameter", fontsize=14)
    plt.title(train_title, fontsize=10)

    plt.grid(color='c', linestyle='-', linewidth=0.2, zorder=-1)
    plt.show()


def plot_dt_risk_with_best_depth(train, best_model, param, data_title):
    """
    Plot desicion tree of the best tree that was found.
    :param train: train set
    :param best_model: the best DT model found at tuning
    :param param: the best mex_depth value found
    :param data_title: the name of the train (by the 3 questions)
    """
    X = train.drop([TARGET], axis=1)
    index_in_order_importance = np.argsort(best_model.feature_importances_)
    for i in range(len(index_in_order_importance)):
        print(best_model.feature_importances_[index_in_order_importance[i]], X.columns.values[index_in_order_importance[i]])
    plt.figure(figsize=(40, 25))
    plot_tree(best_model, feature_names=X.columns.values, filled=True, fontsize=8)
    plt.title(f'Decision tree for {data_title} with max depth {param}')
    plt.show()


def print_features_weights_by_coef(X, model):
    """
    print the feature importance (weights) of the model by increasing order
    :param X: train set without target label
    :param model: the model to print its weights
    """
    index_in_order_importance = np.argsort([abs(x) for x in model.coef_[0]])
    for i in range(len(index_in_order_importance)):
        print(model.coef_[0][index_in_order_importance[i]],
            X.columns.values[index_in_order_importance[i]])


def print_feature_importance(train, model):
    """
    print the feature importance of the model by increasing order
    :param train: train set
    :param model: the model to print its weights
    """
    X = train.drop([TARGET], axis=1)
    index_in_order_importance = np.argsort(model.feature_importances_)
    for i in range(len(index_in_order_importance)):
        print(model.feature_importances_[index_in_order_importance[i]], X.columns.values[index_in_order_importance[i]])


def create_rf_random_grid():
    """
    create a dictionary for tuning random forest by choosing random permutation from it
    :return: the dictionary
    """
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1000, num=11)]
    # Number of features to consider at every split
    max_features = ['sqrt', 'log2']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(1, 100, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of creating the splitting
    criterion = ['gini', 'entropy']
    # Create the random grid
    return {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'criterion': criterion}


def create_rf_grid():
    """
    create a dictionary for final tuning random forest after knows the closet range from the random search
    :return: the dictionary
    """
    all_data_grid = {'max_depth': [10, 15, 20, 25, 30],
                     'max_features': ['sqrt'],
                     'min_samples_leaf': [1, 2, 3],
                     'min_samples_split': [3, 5, 7],
                     'n_estimators': [320, 370, 410],
                     'criterion': ['entropy']}

    without_scores_grid = {'max_depth': [30, 35, 40, 45, 50],
                           'max_features': ['sqrt'],
                           'min_samples_leaf': [1, 2],
                           'min_samples_split': [2, 3],
                           'n_estimators': [770, 820, 870],
                           'criterion': ['entropy']}

    static_data_grid = {'max_depth': [20, 25, 30, 35, 40],
                        'max_features': ['sqrt'],
                        'min_samples_leaf': [1, 2, 4],
                        'min_samples_split': [2, 10, 12],
                        'n_estimators': [350, 400, 450, 500],
                        'criterion': ['gini']}

    return [all_data_grid, without_scores_grid, static_data_grid]


def create_mlp_random_grid():
    """
    create a dictionary for tuning MLP by choosing random permutation from it
    :return: the dictionary
    """
    hidden_layer_sizes = [(64,), (128,), (128, 128), (128, 64, 128), (64, 64), (64, 64, 64), (64, 128, 64),
                          (128, 64, 32)]
    activation = ['logistic', 'tanh', 'relu']
    solver = ['adam', 'sgd']
    alpha = np.logspace(-5, 2, num=8)
    max_iter = [200, 500, 1000, 1500]
    return {'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'max_iter': max_iter}


def main():
    all_years_train = pd.read_csv('Data/train/all_years_final_normal_id.csv')
    train_without_scores = pd.read_csv('Data/train/train_without_scores.csv')
    train_static_match_data = pd.read_csv('Data/train/train_static_match_data.csv')

    data_list_to_fit = [all_years_train, train_without_scores, train_static_match_data]
    data_title = ['all data', 'data without scores of sets', 'only static data of matches']

    # KNN tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation, best_model = model_parameter_tuning(KNeighborsClassifier, range(1, 101, 2),
                                                                           train_data, subtitle=train_title)
        print(f'KNN for {train_title}: \n\t '
              f'Best train k_neigh is: {param_train}, Best validation k_neigh is: {param_validation}')
    print()

    # DT tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation, best_model = model_parameter_tuning(DecisionTreeClassifier, range(1, 101),
                                                                           train_data, subtitle=train_title)
        print(f'DT for {train_title}: \n\t '
              f'Best train max_depth is: {param_train}, Best validation max_depth is: {param_validation}')
        plot_dt_risk_with_best_depth(train_data.drop([TARGET], axis=1), best_model)
    print()

    # LinearSVM tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation, best_model = \
            model_parameter_tuning(LinearSVC, np.logspace(-3, 3, num=100), train_data, is_log_range=True,
                                   subtitle=train_title)
        print(f'LinearSVM for {train_title}: \n\t '
              f'Best train C is: {param_train}, Best validation C is: {param_validation}')
        print_features_weights_by_coef(train_data, param_validation)
    print()

    # SVM poly kernel tune: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        tune_svc_kernel_plot_heatmap(train=train_data, c_range=np.logspace(-4, 4, num=9), kernel='poly',
                                     kernel_range=np.logspace(-4, 4, num=9), train_title=train_title)
    print()

    # SVM rbf kernel tune: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        tune_svc_kernel_plot_heatmap(train=train_data, c_range=np.logspace(-4, 4, num=9), kernel='rbf',
                                     kernel_range=np.logspace(-4, 4, num=9), train_title=train_title)
    print()

    # Random Forest tune
    random_grid = create_rf_random_grid()
    for train_data, train_title in zip(data_list_to_fit, data_title):
        best_params = tune_model_by_random_grid(RandomForestClassifier, train_data, random_grid, train_title)

    # *This part was added after the results of the random grid search were known*
    data_grids = create_rf_grid()
    for train_data, train_title, param_grid in zip(data_list_to_fit, data_title, data_grids):
        best_params = tune_model_by_grid(RandomForestClassifier, train_data, param_grid, train_title)
    print()

    # AdaBoost tune
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation, adaboost_model = \
            model_parameter_tuning(AdaBoostClassifier, range(50, 2001, 50), train_data, subtitle=train_title)
        print(f'AdaBoost for {train_title}: \n\t '
              f'Best train n_estimators is: {param_train}, Best validation n_estimators is: {param_validation}')
        print_feature_importance(train_data, adaboost_model)
    print()

    # Perceptron tune
    for train_data, train_title in zip(data_list_to_fit, data_title):
        tune_perceptron(train_data, train_title)
    print()

    # Multi Layer Perceptron tune
    random_grid = create_mlp_random_grid()
    for train_data, train_title in zip(data_list_to_fit, data_title):
        best_params = tune_model_by_random_grid(MLPClassifier, train_data, create_mlp_random_grid, train_title)


if __name__ == '__main__':
    main()