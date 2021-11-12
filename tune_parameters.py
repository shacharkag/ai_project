import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
KERNELS_PARAM_STR = {'poly': 'degree', 'rbf': 'gamma'}


def get_args_for_classification_model(classification_model, parameter):
    if classification_model is KNeighborsClassifier:
        return {'n_neighbors': parameter}
    elif classification_model is DecisionTreeClassifier:
        return {'criterion': "entropy",
                'max_depth': parameter}
    elif classification_model is LinearSVC:
        return {'C': parameter,
                'max_iter': 5000}


def get_args_for_svc_with_kernel(kernel_type, c_param, kernel_param):
    if kernel_type == 'poly':
        return {'C': c_param, 'kernel': kernel_type, 'degree': kernel_param}
    return {'C': c_param, 'kernel': kernel_type, 'gamma': kernel_param}


def get_args_for_rbf(const_param, param_name, i):
    const_param_name = 'gamma' if param_name == 'C' else 'C'
    return {'kernel': 'rbf', param_name: i, const_param_name: const_param}


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


def tune_svc_kernel_plot_heatmap(train, c_range, kernel, kernel_range, train_title=""):
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
    all_years_train = pd.read_csv('Data/train/all_years_final_normal_id.csv')
    train_without_scores = pd.read_csv('Data/train/train_without_scores.csv')
    train_static_match_data = pd.read_csv('Data/train/train_static_match_data.csv')

    data_list_to_fit = [all_years_train, train_without_scores, train_static_match_data]
    data_title = ['all data', 'data without scores of sets', 'only static data of matches']

    # KNN tuned: all-data, without score data, only static data
    for train_data, train_title in zip(data_list_to_fit, data_title):
        param_train, param_validation = model_prediction(KNeighborsClassifier, range(1, 101, 2), train_data,
                                                         subtitle=train_title)
        print(f'KNN for {train_title}: \n\t '
              f'Best train k_neigh is: {param_train}, Best validation k_neigh is: {param_validation}')
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
        param_train, param_validation = model_prediction(LinearSVC, np.logspace(-3, 3, num=100), train_data,
                                                         is_log_range=True, subtitle=train_title)
        print(f'LinearSVM for {train_title}: \n\t '
              f'Best train C is: {param_train}, Best validation C is: {param_validation}')
        print_linear_svm_weights_features(train_data, param_validation)
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


if __name__ == '__main__':
    main()